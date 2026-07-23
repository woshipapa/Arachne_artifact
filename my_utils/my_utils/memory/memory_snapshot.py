import torch
import torch.distributed as dist
import os
import logging
import time

# --- fallback logger, just in case ---
_default_logger = logging.getLogger("MemorySnapshotter")
if not _default_logger.handlers:
    _default_logger.addHandler(logging.StreamHandler())
    _default_logger.setLevel(logging.INFO)

class MemorySnapshotter:
    """
    A hierarchical (stack-aware) memory snapshot helper that wraps PyTorch's
    _record_memory_history behind a start/stop API.
    
    Flow:
    start("A"):
      start("B"):
      stop("B"): -> immediately writes a "B.pt" snapshot (holding the A+B history)
    stop("A"):  -> immediately writes an "A.pt" snapshot (holding the A+B history)
    
    (Note: "A.pt" contains everything "B.pt" does. Diff the two files to isolate what
    """
    def __init__(self, save_dir="memory_snapshots"):
        self.is_enabled = os.environ.get("ENABLE_MEMORY_SNAPSHOT", "0") == "1"
        self.save_dir = save_dir
        self.logger = _default_logger
        
        if self.is_enabled:
            if not (torch.cuda.is_available()):
                self.logger.warning(
                    "ENABLE_MEMORY_SNAPSHOT=1 but torch.cuda.is_available() is False. Disabling."
                )
                self.is_enabled = False
            else:
                os.makedirs(self.save_dir, exist_ok=True)
                self.logger.info(
                    f"MemorySnapshotter is ENABLED. "
                    f"snapshots will be written to: {self.save_dir}"
                )
        else:
            self.logger.info("MemorySnapshotter is DISABLED.")

        # the core of this class: a stack, mirroring MyTimer V2
        self._active_names = [] 

    def set_logger(self, logger):
        """Inject an external logger, as MyTimer does."""
        self.logger = logger
    
    def start(self, name: str, max_entries=100000):
        """
        Begin (or continue) recording memory history.
        On the first 'start' (empty stack) the global recorder is switched on.
        """
        if not self.is_enabled:
            return
        
        try:
            if not self._active_names:
                # empty stack: this is the root call
                # start and *reset* the history, then record globally
                torch.cuda.memory._record_memory_history(
                    enabled=True
                )
                self.logger.info(f"[MemSnapshot] START-ROOT: '{name}' (history started and reset)")
            else:
                # already nested; just make sure the recorder is still running
                # torch.cuda.memory._record_memory_history(enabled=True)
                self.logger.info(f"[MemSnapshot] START-CHILD: '{name}'")
            
            # push onto the stack
            self._active_names.append(name)
            
        except Exception as e:
            self.logger.error(f"[MemSnapshot] FAILED to start recording for '{name}': {e}")
            self.is_enabled = False # disable to avoid log spam

    def stop(self, name: str):
        """
        Stop one timer and immediately dump the history *so far* to a .pt file.
        On the last 'stop' (stack becomes empty) the global recorder is switched off.
        """
        if not self.is_enabled:
            return

        # --- same stack-checking logic as MyTimer V2 ---
        if not self._active_names or self._active_names[-1] != name:
            self.logger.warning(
                f"[MemSnapshot] Mismatched STOP call! "
                f"Expected '{self._active_names[-1] if self._active_names else 'None'}' "
                f"but got '{name}'."
            )
            # robustness: pop it if we can find it at all
            if name not in self._active_names:
                self.logger.error(f"[MemSnapshot] FAILED to stop '{name}'. Not on the active stack.")
                return
            # pop children until we reach it
            while self._active_names.pop() != name:
                pass
        else:
            # match: pop normally
            self._active_names.pop()
        # --- end of stack checking ---

        try:
            # 1. build the filename
            time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            rank = torch.distributed.get_rank() if dist.is_initialized() else 0
            
            # format: [save_dir]/[name]__[time]__rank[rank].pt
            filename = os.path.join(
                self.save_dir, 
                f"{name.replace(' ', '_').replace('/', '-')}__{time_str}__rank{rank}.pt"
            )
            
            # 2. the core step: dump the entire history *as of now*
            torch.cuda.memory._dump_snapshot(filename)
            
            self.logger.info(f"[MemSnapshot] STOP: '{name}'. "
                             f"snapshot (holding all history *so far*) "
                             f"written to: {filename}")
            
            # 3. key detail: if the stack is *now* empty, stop and reset the history
            if not self._active_names:
                self.logger.info(f"[MemSnapshot] STOP-ROOT: "
                                 f"all snapshots stopped. Resetting memory history.")
                torch.cuda.memory._record_memory_history(enabled=None) # reset

        except Exception as e:
            self.logger.error(f"[MemSnapshot] FAILED to dump snapshot for '{name}': {e}")

# --- no-op version ---
class NoOpMemorySnapshotter:
    """A 'do nothing' version to match NoOpTimer."""
    def set_logger(self, logger): pass
    def start(self, name: str, max_entries=100000): pass
    def stop(self, name: str): pass

# --- the global instance ---
# call sites do `from my_memory_utils import global_snapshotter`
if os.environ.get("ENABLE_MEMORY_SNAPSHOT", "0") == "1":
    global_snapshotter = MemorySnapshotter(save_dir="memory_snapshots")
else:
    global_snapshotter = NoOpMemorySnapshotter()
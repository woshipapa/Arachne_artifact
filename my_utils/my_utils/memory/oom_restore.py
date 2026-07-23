import datetime
from datetime import timedelta
import torch
def set_oom_flag():
    """Set the OOM flag in the distributed store."""
    try:
        if torch.distributed.is_initialized():
            # get the default TCPStore
            store = torch.distributed.distributed_c10d._get_default_store()
            if store is not None:
                # set a key; any non-empty value will do
                store.set("GLOBAL_OOM_TRIGGERED", "1")
                print("[Signal] Broadcast OOM signal to the cluster.")
    except Exception as e:
        print(f"[Warning] Failed to broadcast the OOM signal: {e}")

def check_oom_flag():
    """Check whether any rank in the cluster hit an OOM."""
    try:
        if torch.distributed.is_initialized():
            store = torch.distributed.distributed_c10d._get_default_store()
            if store is not None:
                # check whether the key exists
                # keep the timeout short so we do not block forever
                try:
                    val = store.get("GLOBAL_OOM_TRIGGERED")
                    if val == b"1":
                        return True
                except:
                    return False
    except:
        pass
    return False
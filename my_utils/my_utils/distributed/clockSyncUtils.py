import time
import torch
import torch.distributed as dist

class ClockSynchronizer:
    """
    Distributed clock synchronisation following NTP (Network Time Protocol) logic.
    Measures the system-clock offset between a producer and a consumer host.
    """
    
    @staticmethod
    def sync(is_server: bool, peer_rank: int, group=None):
        """
        Run the synchronisation.
        
        Args:
            is_server (bool): 
                True  = reference side (typically the producer); offset is 0.
                False = follower side (typically the consumer); computes its offset to the server.
            peer_rank (int): the peer's rank within the group.
            group: process group, default None (the global group).
            
        Returns:
            float: time_offset (Server_Time - Client_Time)。
                   Adding it to the consumer's timer yields the producer's clock.
        """
        # 1. Hard barrier: make sure both sides have reached the start line.
        # This removes the spurious latency of "consumer sends first while the producer is still busy".
        dist.barrier(group=group)
        
        # data buffers (float64/double, to preserve microsecond precision)
        # the NCCL backend generally requires the tensor to live on CUDA
        tensor_buffer = torch.zeros(2, dtype=torch.float64, device=torch.cuda.current_device())
        
        if is_server:
            # === Server Logic (Producer) ===
            
            # A. wait for the client's ping
            # the producer is already past the barrier, so the arrival instant is T2
            dist.recv(tensor_buffer, src=peer_rank, group=group)
            t2 = time.time() # T2: Server Receive Timestamp
            
            # B. record the reply time T3
            t3 = time.time() # T3: Server Send Timestamp
            
            # C. pack T2 and T3 and send them back
            tensor_buffer[0] = t2
            tensor_buffer[1] = t3
            dist.send(tensor_buffer, dst=peer_rank, group=group)
            
            print(f"[ClockSync] Server ready. Reference time provided.")
            return 0.0

        else:
            # === Client Logic (Consumer) ===
            
            # A. send the ping
            t1 = time.time() # T1: Client Send Timestamp
            dist.send(tensor_buffer, dst=peer_rank, group=group) # empty packet, triggers T2
            
            # B. receive the pong (carrying T2 and T3)
            dist.recv(tensor_buffer, src=peer_rank, group=group)
            t4 = time.time() # T4: Client Receive Timestamp
            
            t2 = tensor_buffer[0].item()
            t3 = tensor_buffer[1].item()
            
            # === core NTP computation ===
            # RTT = total elapsed time - the server's processing time
            rtt = (t4 - t1) - (t3 - t2)
            latency = rtt / 2.0
            
            # Offset = T_server - T_client
            # T_server_at_t1 = t2 - latency
            offset = (t2 - latency) - t1
            
            print("="*40)
            print(f"[ClockSync] Client Synchronization Result:")
            print(f"  > RTT     : {rtt*1000:.3f} ms")
            print(f"  > Latency : {latency*1000:.3f} ms (One-way)")
            print(f"  > Offset  : {offset:.6f} s")
            print("="*40)
            
            return offset
        


import time
import socket
import struct
import pickle

class SocketClockSynchronizer:
    def __init__(self, port=12345):
        self.port = port
        self.sock = None
        self.conn = None

    def sync(self, is_server: bool, peer_ip: str = '0.0.0.0'):
        """
        Synchronise over a raw TCP socket, without torch.distributed.
        """
        offset = 0.0
        
        try:
            # ==========================================
            # 1. establish the TCP connection (the handshake)
            # ==========================================
            if is_server:
                # producer acts as the server
                server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server_sock.bind(('0.0.0.0', self.port))
                server_sock.listen(1)
                print(f"[SocketSync] Listening on port {self.port}...")
                
                self.conn, addr = server_sock.accept()
                print(f"[SocketSync] Connected by {addr}")
                
            else:
                # consumer acts as the client
                # brief sleep so the server is up first
                time.sleep(2) 
                print(f"[SocketSync] Connecting to {peer_ip}:{self.port}...")
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # connect, with retries
                for i in range(10):
                    try:
                        self.sock.connect((peer_ip, self.port))
                        break
                    except ConnectionRefusedError:
                        print(f"Retrying connection... ({i+1}/10)")
                        time.sleep(1)
                self.conn = self.sock

            # ==========================================
            # 2. simple barrier (both sides ready)
            # ==========================================
            # each side sends one byte and waits for the other's
            self.conn.sendall(b'1') # I am here
            self.conn.recv(1)       # wait for you
            
            # both sides are now past the barrier and the CPU is relatively idle
            
            # ==========================================
            # 3. NTP exchange
            # ==========================================
            if is_server:
                # --- Server Logic ---
                # A. wait for the ping
                data = self.conn.recv(1024) # blocking
                t2 = time.time()            # T2: Recv Time
                
                # B. send the pong (T2, T3)
                t3 = time.time()            # T3: Send Time
                payload = struct.pack('!dd', t2, t3) # double (8 bytes) * 2
                self.conn.sendall(payload)
                
                return 0.0
                
            else:
                # --- Client Logic ---
                # A. send the ping
                t1 = time.time()            # T1: Send Time
                self.conn.sendall(b'PING')
                
                # B. receive the pong
                data = self.conn.recv(1024)
                t4 = time.time()            # T4: Recv Time
                
                if len(data) == 16:
                    t2, t3 = struct.unpack('!dd', data)
                    
                    rtt = (t4 - t1) - (t3 - t2)
                    latency = rtt / 2.0
                    offset = (t2 - latency) - t1
                    
                    print(f"[SocketSync] RTT: {rtt*1000:.3f}ms, Offset: {offset:.6f}s")
                    return offset
                else:
                    print("Error: Invalid packet size")
                    return 0.0
                    
        finally:
            # tear down the connection
            if self.conn: self.conn.close()
            if is_server and 'server_sock' in locals(): server_sock.close()
    # def _broker_worker_loop(self):
    #     """
    #     """
    #     while True:
    #         active_senders = self.broker_task_queue.get()
    #         active_senders = [8]
    #             self.logger.info("Broker main loop: Received stop signal. Exiting.")
    #             break
            
    #         # ==========================================================
    #         # ==========================================================
    #         remote_senders = [r for r in active_senders if r != self.rank]
    #         # ==========================================================
            
    #         if not remote_senders:
    #             self.logger.info(f"Broker loop: No remote senders to listen to for this iteration. Waiting for next iteration.")
    #             self.broker_task_queue.task_done()
    #             continue

    #         self.logger.info(f"Broker loop: New iteration started. Listening only to REMOTE ranks: {remote_senders}")

    #         requests = {}
    #         for r in remote_senders:
    #             size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
    #             req = dist.recv(tensor=size_tensor, src=r, group=self.cpu_service_group, tag=self.SERVICE_TAG)
    #             requests[r] = (req, size_tensor)
            
    #         remaining_ranks = set(requests.keys())
    #         while remaining_ranks:
    #             for src_rank in list(remaining_ranks):
    #                 if src_rank not in requests: continue
                    
    #                 req, size_tensor = requests[src_rank]
    #                 # if req.is_completed():
    #                 self.logger.info(f"Broker main loop: Detected a completed request from remote Rank {src_rank}.")
    #                 handler_thread = threading.Thread(target=self._handle_request, args=(src_rank, size_tensor))
    #                 handler_thread.start()
                    
    #                 remaining_ranks.remove(src_rank)
    #                 del requests[src_rank]

    #             time.sleep(0.001)

    #         self.logger.info(f"Broker loop: Finished listening for all expected remote requests for this iteration.")
    #         self.broker_task_queue.task_done()


    # def _broker_worker_loop(self):
    #         """
    #         """
    #         while True:
    #             comm_work_list = self.broker_task_queue.get()
    #             if comm_work_list is None: break

    #             comm_counts = Counter(comm_work_list)
    #             remote_comm_counts = {r: c for r, c in comm_counts.items() if r != self.rank}
    #             if not remote_comm_counts:
    #                 self.logger.info(f"Broker loop: No remote requests to handle for this iteration.")
    #                 self.broker_task_queue.task_done()
    #                 continue

    #             self.logger.info(f"Broker loop: New iteration started. Expecting requests: {remote_comm_counts}")
    #             num_expected_requests = sum(remote_comm_counts.values())

    #             for i in range(num_expected_requests):
    #                 try:
    #                     size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
    #                     sender_rank = dist.recv(tensor=size_tensor, src=None, group=self.cpu_service_group, tag=self.SERVICE_TAG)
                        
    #                     if sender_rank not in remote_comm_counts or remote_comm_counts[sender_rank] == 0:
    #                         self.logger.warning(f"Broker: Received an unexpected or extra message from Rank {sender_rank}.")
    #                         continue

    #                     self.logger.info(f"Broker: Received request #{i+1}/{num_expected_requests} from Rank {sender_rank}.")
                        
    #                     handler_thread = threading.Thread(target=self._handle_request, args=(sender_rank, size_tensor))
    #                     handler_thread.start()
                        
    #                     remote_comm_counts[sender_rank] -= 1

    #                 except Exception:
    #                     self.logger.error("Broker main loop encountered an exception while receiving.", exc_info=True)
                
    #             self.logger.info(f"Broker loop: Finished all {num_expected_requests} expected requests for this iteration.")
    #             self.broker_task_queue.task_done()
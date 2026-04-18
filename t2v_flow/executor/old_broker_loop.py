    # def _broker_worker_loop(self):
    #     """
    #     【智能 Broker 最终版】按计划只监听本轮活跃的、远程的 rank。
    #     """
    #     while True:
    #         # 1. 从主线程的队列中获取新一轮迭代的监听列表
    #         # .get() 会在此阻塞，直到下一轮迭代开始
    #         active_senders = self.broker_task_queue.get()
    #         active_senders = [8]
    #         if active_senders is None:  # 收到停止信号
    #             self.logger.info("Broker main loop: Received stop signal. Exiting.")
    #             break
            
    #         # ==========================================================
    #         # ## 核心修复：从监听列表中排除自己 ##
    #         # ==========================================================
    #         remote_senders = [r for r in active_senders if r != self.rank]
    #         # ==========================================================
            
    #         if not remote_senders:
    #             self.logger.info(f"Broker loop: No remote senders to listen to for this iteration. Waiting for next iteration.")
    #             self.broker_task_queue.task_done()
    #             continue

    #         self.logger.info(f"Broker loop: New iteration started. Listening only to REMOTE ranks: {remote_senders}")

    #         # 2. 只为活跃的、远程的 rank 创建 irecv 监听器
    #         requests = {}
    #         for r in remote_senders:
    #             size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
    #             req = dist.recv(tensor=size_tensor, src=r, group=self.cpu_service_group, tag=self.SERVICE_TAG)
    #             requests[r] = (req, size_tensor)
            
    #         # 3. 监听并处理本轮的所有预期请求
    #         remaining_ranks = set(requests.keys())
    #         while remaining_ranks:
    #             for src_rank in list(remaining_ranks):
    #                 # 确保在循环中检查 key 是否仍然存在
    #                 if src_rank not in requests: continue
                    
    #                 req, size_tensor = requests[src_rank]
    #                 # if req.is_completed():
    #                 self.logger.info(f"Broker main loop: Detected a completed request from remote Rank {src_rank}.")
    #                 # 将完整的请求处理分派给子线程
    #                 handler_thread = threading.Thread(target=self._handle_request, args=(src_rank, size_tensor))
    #                 handler_thread.start()
                    
    #                 # 这个 rank 的本轮请求已开始处理，将其移出监听集合
    #                 remaining_ranks.remove(src_rank)
    #                 del requests[src_rank]

    #             time.sleep(0.001)

    #         self.logger.info(f"Broker loop: Finished listening for all expected remote requests for this iteration.")
    #         self.broker_task_queue.task_done()



    # def _broker_worker_loop(self):
    #         """
    #         【最终健壮版 Broker】使用阻塞式 recv(src=None) 和计数器来串行处理所有预期请求。
    #         这个模型简单、稳定，能正确处理乱序到达和重复 rank 的问题。
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

    #             # 循环处理本轮预期的请求总数
    #             for i in range(num_expected_requests):
    #                 try:
    #                     size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
    #                     # 核心：阻塞等待，直到 ANY rank 发送消息，并返回 sender_rank
    #                     sender_rank = dist.recv(tensor=size_tensor, src=None, group=self.cpu_service_group, tag=self.SERVICE_TAG)
                        
    #                     if sender_rank not in remote_comm_counts or remote_comm_counts[sender_rank] == 0:
    #                         self.logger.warning(f"Broker: Received an unexpected or extra message from Rank {sender_rank}.")
    #                         # 如果收到了预期之外的消息，简单地忽略它并等待下一个
    #                         # 重新进入 for 循环的下一次迭代
    #                         continue

    #                     self.logger.info(f"Broker: Received request #{i+1}/{num_expected_requests} from Rank {sender_rank}.")
                        
    #                     # 将请求分派给子线程处理
    #                     handler_thread = threading.Thread(target=self._handle_request, args=(sender_rank, size_tensor))
    #                     handler_thread.start()
                        
    #                     # 将该 rank 的预期请求数减 1
    #                     remote_comm_counts[sender_rank] -= 1

    #                 except Exception:
    #                     self.logger.error("Broker main loop encountered an exception while receiving.", exc_info=True)
                
    #             self.logger.info(f"Broker loop: Finished all {num_expected_requests} expected requests for this iteration.")
    #             self.broker_task_queue.task_done()
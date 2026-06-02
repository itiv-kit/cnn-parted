from dataclasses import dataclass

from framework.model.onnx_model import OnnxModel
from framework.scheduling import SystemSchedule

from loguru import logger

@dataclass
class DeadlineStats:
    success: int = 0
    partial_violation: int = 0
    end_violation: int = 0
    epoch_success: int = 0

#@dataclass
#class DeadlineStats:
#    frames_valid: list[bool]
#    @property
#    def epoch_valid(self):
#        return self.frames_valid[-1]
#    @property
#    def partial_valid(self):
#        # Only the last frame is invalid
#        return not self.epoch_valid and all(self.frames_valid[:-1])
#    @property
#    def successful(self):
#        return self.epoch_valid


def deadline_evaluation(models: dict[str, OnnxModel], 
                        results: list[SystemSchedule], 
                        invalid_results: list[SystemSchedule]):

        deadline_stats = {network: DeadlineStats() for network in models.keys()}
        all_partial_succeeded = []
        all_fully_succeeded = []
        for schedule in results + invalid_results:
            network_frame_endtimes = schedule.network_finish_times
            partial_succ = True
            fully_succ = True
            for nw, frame_endtimes in network_frame_endtimes.items():
                
                # Check if every single frame finishes with the deadline
                for frame_endtime, deadline in zip(frame_endtimes, models[nw].deadlines, strict=True):
                    if frame_endtime > deadline:
                        deadline_stats[nw].partial_violation += 1
                        fully_succ = False
                        break
                
                if frame_endtimes[-1] > 1000 and models[nw].has_frame_constraint:
                    # Check if the last frame is finished before 1s elapsed
                    partial_succ = False
                    fully_succ = False
                    deadline_stats[nw].end_violation += 1
                else:
                    # If the last frame is successful, we count it as a success
                    # This is a relaxed definition
                    partial_succ = True
                    deadline_stats[nw].epoch_success += 1

            all_fully_succeeded.append(fully_succ)
            all_partial_succeeded.append(partial_succ)
                
        num_all_succ = all_partial_succeeded.count(True)
        num_sched = len(all_partial_succeeded)
        logger.info(f"All deadlines were wet in {num_all_succ}/{num_sched} cases ({num_all_succ/num_sched*100:.2f}%)")
        for nw, stats in deadline_stats.items():
            if models[nw].has_constraints:
                logger.info(f"End-frame success for {nw} in {stats.epoch_success}/{num_sched} cases ({stats.end_success/(num_sched)*100:.2f}%)")
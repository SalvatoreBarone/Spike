import torch, snntorch, os, matplotlib.pyplot as plt
from tqdm import tqdm, trange
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group

def ddp_setup(rank, world_size):
    """
    First, before initializing the group process, call set_device, which sets the default GPU for each process.
    This is important to prevent hangs or excessive memory utilization on GPU:0
    The process group can be initialized by TCP (default) or from a shared file-system. 
    init_process_group initializes the distributed process group.

    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class SimpleDDPTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        forward_pass_fn : callable,
        train_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader,
        loss_fn: snntorch.functional.loss.LossFunctions,
        acc_fn : callable,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        gpu_id: int,
        save_every: int,
        workdir : str,
        snapshot_path: str = "snapshot.pth",
        model_name : str = "model.pth"
    ) -> None:
        self.model = model.to(gpu_id)
        self.forward_pass_fn = forward_pass_fn
        self.train_data = train_data
        self.test_data = test_data
        self.loss_fn = loss_fn
        self.acc_fn = acc_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gpu_id = gpu_id
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.workdir = workdir
        self.snapshot_path = f"{self.workdir}/{snapshot_path}"
        self.model_path = f"{self.workdir}/{model_name}"
        self.epochs_run = 0
        if os.path.exists(self.snapshot_path):
            self._load_snapshot()

    def load(self):
        self.model.module.load_state_dict(torch.load(self.model_path, weights_only=True))
        print(f"Model loaded from {self.model_path}")

    def save(self):
        torch.save(self.model.module.state_dict(), self.model_path)
        print(f"Final model saved at {self.model_path}")

    def train(self, max_epochs: int):
        torch.autograd.set_detect_anomaly(True)
        self.loss_hist = [float("inf")]
        self.train_acc_hist = [0]
        self.test_acc_hist = [0]
        
        if self.gpu_id == 0:
            pbar = tqdm(total=max_epochs, desc = "Training: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}")
            if self.epochs_run > 0:
                print(f"Resuming training from snapshot at Epoch {self.epochs_run}.")
                pbar.update(self.epochs_run)
            
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            self._test()
            
            if self.gpu_id == 0:
                pbar.update(1)
                tqdm.write(
                    f"\nEpoch: {epoch+1}/{max_epochs}.\n"
                    f"\tTraining loss: {self.loss_hist[-1]:.4f}.\n"
                    f"\tTraining accuracy: {self.train_acc_hist[-1]*100:.2f}%.\n"
                    f"\tTest accuracy: {self.test_acc_hist[-1]*100:.2f}%"
                )
                
            if self.gpu_id == 0 and epoch+1 % self.save_every == 0:
                self._save_snapshot(epoch)
                
        if self.gpu_id == 0:
            self._plot()
            self.save()
            
        if os.path.exists(self.snapshot_path):
            os.remove(self.snapshot_path)

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        
        if self.gpu_id == 0:
            pbar = tqdm(total = len(self.train_data), desc = f"Epoch {epoch+1}, batches: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}")
            
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
            
            if self.gpu_id == 0:
                pbar.update(1)
            

    def _run_batch(self, source, targets):
        self.model.module.train()
        # propagating one batch through the network and evaluating loss
        recs = self.forward_pass_fn(self.model.module, source)
        loss_val = self.loss_fn(recs[0], targets)
         # Gradient calculation + weight update
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()
        # Store loss history for future plotting
        self.loss_hist.append(loss_val.item())
        self.train_acc_hist.append(self.acc_fn(recs[0], targets))
        
        # Step the scheduler at the end of each epoch
        if self.scheduler is not None:
            self.scheduler.step(loss_val)
            if self.gpu_id == 0:
                tqdm.write(f"\tLearning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            


    def _test(self):
        with torch.no_grad():
            total = 0
            acc = 0
            self.model.module.eval()
            
            if self.gpu_id == 0:
                pbar = tqdm(total = len(self.test_data), desc = f"Computing test accuracy: ", leave = False, bar_format="{desc:30} {percentage:3.0f}% |{bar:40}{r_bar}{bar:-10b}")
        
            for source, targets in iter(self.test_data):
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                recs = self.forward_pass_fn(self.model, source)
                acc += self.acc_fn(recs[0], targets) * recs[0].size(1)
                total += recs[0].size(1)
                
                if self.gpu_id == 0:
                    pbar.update(1)
                    
        self.test_acc_hist.append(acc / total)

    def _plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(18,4))
        # Plot Train Accuracy
        axes[0].plot(self.train_acc_hist)
        axes[0].set_title("Train Set Accuracy")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Accuracy")
        # Plot Test Accuracy
        axes[1].plot(self.test_acc_hist)
        axes[1].set_title("Test Set Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        # Plot Training Loss
        axes[2].plot(self.loss_hist)
        axes[2].set_title("Loss History")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Loss")
        plt.title("Training History")
        plt.savefig(f"{self.workdir}/training_history.pdf")

    def _save_snapshot(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _load_snapshot(self):
        snapshot = torch.load(self.snapshot_path)
        self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot {self.snapshot_path} at Epoch {self.epochs_run}")
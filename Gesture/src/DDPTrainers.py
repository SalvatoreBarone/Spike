import torch, snntorch, os, matplotlib.pyplot as plt
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
        forward_pass_fn : function,
        train_data: torch.utils.data.DataLoader,
        test_data: torch.utils.data.DataLoader,
        loss_fn: snntorch.functional.LossFunctions,
        acc_fn : snntorch.functional.Accuracy,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        gpu_id: int,
        save_every: int,
        output_dir_path : str,
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
        self.outdir = output_dir_path,
        self.snapshot_path = f"{output_dir_path}/{snapshot_path}"
        self.model_path = f"{self.outdir}/{model_name}"
        self.epochs_run = 0
        if os.path.exists(self.snapshot_path):
            self._load_snapshot()

    def load(self):
        self.model.module.load_state_dict(torch.load(self.model_path, weights_only=True))

    def save(self):
        torch.save(self.model.module.state_dict(), self.model_path)

    def train(self, max_epochs: int):
        torch.autograd.set_detect_anomaly(True)
        self.loss_hist = [float("inf")]
        self.train_acc_hist = [0]
        self.test_acc_hist = [0]
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            self._test()
            if self.gpu_id == 0:
                print(
                    f"\nEpoch: {epoch}/{max_epochs}.\n"
                    f"\tTraining loss: {self.loss_hist[-1]}.\n"
                    f"\tTraining accuracy: {self.train_acc_hist[-1]*100}%.\n"
                    f"\tTest accuracy: {self.test_acc_hist[-1]*100}%"
                )
            if self.gpu_id == 0 and epoch+1 % self.save_every == 0:
                self._save_snapshot(epoch)
        if self.gpu_id == 0:
            self._plot()
            self.save()
            os.remove(self.snapshot_path)

    def _run_epoch(self, epoch):
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
            

    def _run_batch(self, source, targets):
        self.model.module.train()
        self.optimizer.zero_grad()
        spk_rec, _ = self.forward_pass_fn(self.model.module, source)
        loss_val = self.loss_fn(spk_rec, targets)
        loss_val.backward()
        self.optimizer.step()
        
        self.loss_hist.append(loss_val.item())
        self.train_acc_hist.append(self.acc_fn(spk_rec, targets))
        if self.scheduler is not None:
            # Step the scheduler at the end of each epoch
            self.scheduler.step(loss_val)
            if self.gpu_id == 0:
                print(f"\tLearning Rate: {self.optimizer.param_groups[0]['lr']}")
        

    def _test(self):
        with torch.no_grad():
            total = 0
            acc = 0
            self.model.module.eval()
            for source, targets in iter(self.test_data):
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                spk_rec, _ = self.forward_pass_fn(self.model, source)
                acc += self.acc_fn(spk_rec, targets) * spk_rec.size(1)
                total += spk_rec.size(1)
        self.test_acc_hist.append(acc / total)

    def _plot(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        ln1 = ax1.plot(self.loss_hist, label="Train Loss", color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.set_ylabel("Loss", color='tab:red')
        
        ln2 = ax2.plot(self.train_acc_hist, label="Train Accuracy", color='tab:orange')
        ln3 =ax2.plot(self.test_acc_hist, label="Test Accuracy", color='tab:green')
        ax2.set_ylabel("Accuracy")
        
        lns = ln1 + ln2 + ln3
        labs = [l.get_label() for l in lns ]
        plt.legend(lns, labs)
        
        plt.xlabel("Epoch")
        plt.title("Training History")
        plt.savefig(f"{self.outdir}/training_history.pdf")

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
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
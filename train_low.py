from cvpiclr.models.detect import DetectTrainer, LowNet, MidNet, HighNet
import hyp

model = LowNet().to(hyp.device)

trainer = DetectTrainer(model, hyp.lownet_path, hyp.detection_dataset_dir, hyp.device)
trainer.train(200)

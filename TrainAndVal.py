import torch
from Checkpointing import *

def train_part(epochs, model, train_dl, optimizer, lr_scheduler, acc_fn, loss_fn, writer, DEVICE):
    train_loss = 0
    train_acc = 0

    model.train()

    for batch, data in enumerate(train_dl):
        print(f"Batch number: {batch}, Type of data: {type(data)}, Length of data: {len(data)}")
        if isinstance(data, tuple) and len(data) == 2:
            image_batch, label_batch =data
            # Prepare data and label batches
            image_batch  = image_batch.to(DEVICE)
            label_batch  = label_batch.to(DEVICE)
            output = model(image_batch)
            
            loss =  loss_fn(output,label_batch)
            train_loss += loss
            train_acc += acc_fn(output, label_batch) 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            print(f"Data format error in batch {batch}")
        
    train_loss /= len(train_dl)
    train_acc  /= len(train_dl)
    
    
    print(f"Train loss: {train_loss:.5f} | Train acc: {train_acc:.2f}% | LR: {lr_scheduler.get_last_lr()[0]:.5f}")
    
    writer.add_scalar(f"Loss/train", train_loss,epochs)
    writer.add_scalar(f"Acc/train", train_acc,epochs)

    if(lr_scheduler.get_last_lr()[0]>=0.00001):
        lr_scheduler.step()

def val_part(epochs, model, val_dl, early_stopper, acc_fn, loss_fn, writer, DEVICE):
 
    val_loss = 0 
    val_acc = 0
  
    model.eval()

    with torch.inference_mode():
        for image_batch, label_batch in val_dl:
            image_batch  = image_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            output = model(image_batch)

            val_loss += loss_fn(output, label_batch)
            val_acc += acc_fn(output, label_batch)
      
        val_loss /= len(val_dl)
        val_acc /= len(val_dl)

       
        print(f"Validation loss: {val_loss:.5f} | Validation acc: {val_acc:.2f}\n")

        writer.add_scalar(f"Loss/val", val_loss,epochs)
        writer.add_scalar(f"Acc/val", val_acc,epochs)

        if early_stopper.should_stop(val_acc):
            print(f"\nValidation accuracy has not improved for {early_stopper.epoch_counter} epoch, aborting...")
            return 0
        else:
            if early_stopper.epoch_counter > 0:
                print (f"Epochs without improvement: {early_stopper.epoch_counter}\n")
            return 1

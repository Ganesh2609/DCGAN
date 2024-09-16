import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_generator(generator:torch.nn.Module,
                    discriminator:torch.nn.Module,
                    loss_fn:torch.nn.Module,
                    optimizer:torch.optim.Optimizer,
                    BATCH_SIZE:int,
                    Z_DIM:int,
                    device:torch.device):
    
    generator.train()
    discriminator.train()
    
    noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
    true_labels = torch.ones(BATCH_SIZE, 1, device=device)
    
    # Generating loss
    fake_img = generator(noise)
    pred_probs = discriminator(fake_img)
    loss = loss_fn(pred_probs, true_labels)
    
    # Updation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    del noise, true_labels, fake_img, pred_probs
    return loss.item()


def train_discriminator(generator:torch.nn.Module,
                        discriminator:torch.nn.Module,
                        batch:tuple,
                        loss_fn:torch.nn.Module,
                        optimizer:torch.optim.Optimizer,
                        BATCH_SIZE:int,
                        Z_DIM:int,
                        device:torch.device):
    
    generator.train()
    discriminator.train()
    
    real_img, _ = batch
    real_img = real_img.to(device)
    real_label = torch.ones(real_img.shape[0], 1, device=device)
    
    noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
    fake_img = generator(noise)
    fake_label = torch.zeros(BATCH_SIZE, 1, device=device)
    
    # Generating loss
    real_probs = discriminator(real_img)
    real_score = torch.mean(torch.sigmoid(real_probs)).item()
    real_loss = loss_fn(real_probs, real_label)
    
    fake_probs = discriminator(fake_img)
    fake_score = torch.mean(torch.sigmoid(fake_probs)).item()
    fake_loss = loss_fn(fake_probs, fake_label)
    
    loss = (real_loss + fake_loss)/2
    
    # Updation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    del real_img, real_label, noise, fake_img, fake_label, real_probs, fake_probs
    return loss.item(), real_score, fake_score   


def train_models(generator:torch.nn.Module,
                 discriminator:torch.nn.Module,
                 dataloader:torch.utils.data.DataLoader,
                 loss_fn:torch.nn.Module,
                 gen_optimizer:torch.optim.Optimizer,
                 disc_optimizer:torch.optim.Optimizer,
                 BATCH_SIZE:int,
                 Z_DIM:int,
                 NUM_EPOCHS:int,
                 device:torch.device,
                 gen_path:str=None,
                 disc_path:str=None,
                 result_path:str=None):
    
    results = {
        'Generator Loss' : [],
        'Discriminator Loss' : []
    }
    
    
    for epoch in range(1, NUM_EPOCHS+1):

        disc_loss = 0   
        gen_loss = 0   
        with tqdm(enumerate(dataloader), total=len(dataloader)) as t:
            for i, batch in t:
                disc_batch_loss, real_score, fake_score = train_discriminator(generator=generator, 
                                                                            discriminator=discriminator, 
                                                                            batch=batch, 
                                                                            loss_fn=loss_fn, 
                                                                            optimizer=disc_optimizer, 
                                                                            BATCH_SIZE=BATCH_SIZE, 
                                                                            Z_DIM=Z_DIM, 
                                                                            device=device)
                
                gen_batch_loss = train_generator(generator=generator,
                                                 discriminator=discriminator,
                                                 loss_fn=loss_fn,
                                                 optimizer=gen_optimizer,
                                                 BATCH_SIZE=BATCH_SIZE,
                                                 Z_DIM=Z_DIM,
                                                 device=device)
                
                disc_loss += disc_batch_loss
                gen_loss += gen_batch_loss

                t.set_description(f'Epoch [{epoch}/{NUM_EPOCHS}] ')
                t.set_postfix({
                    'Gen Batch Loss' : gen_batch_loss,
                    'Gen Loss' : gen_loss/(i+1),
                    'Disc Batch Loss' : disc_batch_loss,
                    'Disc Loss' : disc_loss/(i+1),
                    'Real' : real_score,
                    'Fake' : fake_score
                })
                
                if gen_path:
                    torch.save(obj=generator.state_dict(), f=gen_path)
                if disc_path:
                    torch.save(obj=discriminator.state_dict(), f=disc_path)
                
                # Save results every 100 batches
                if i % 100 == 0 and result_path:
                    RESULT_SAVE_NAME = result_path + f'/Epoch_{epoch}.png'
                    generator.eval()
                    discriminator.eval()
                    with torch.inference_mode():
                        noise = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
                        fake_img = generator(noise).cpu()
                    fake_img = torch.clamp((fake_img + 1) / 2, 0, 1)
                    
                    fig, ax = plt.subplots(4, 8, figsize=(15, 8))
                    for i, ax in enumerate(ax.flat):
                        ax.imshow(fake_img[i].permute(1,2,0))
                        ax.axis(False);
                    plt.tight_layout()
                    plt.savefig(RESULT_SAVE_NAME)
                    plt.close(fig)
                
        results['Generator Loss'].append(gen_loss/len(dataloader))
        results['Discriminator Loss'].append(disc_loss/len(dataloader))
        
    return results
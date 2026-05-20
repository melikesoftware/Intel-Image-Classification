from pathlib import Path
import os


from torchvision import datasets
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import  torchvision
import torch
from torchinfo import summary
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

fruit_path=Path("images/")
image_path=fruit_path/"ıntel image"

print(image_path)

def check_data(dir_path):
    for dirpath,dirnames,filenames in os.walk(dir_path):
        print(f"# of direcitories: {len(dirnames)} and {len(filenames)} images in {dirpath}")


check_data(image_path)

train_directory=image_path/"seg_train"/"seg_train"
test_directory=image_path/"seg_test"/"seg_test"
print(train_directory)
print(test_directory)

weights=torchvision.models.ResNet18_Weights.DEFAULT
data_transform=weights.transforms()
print(data_transform)




train_dataset=datasets.ImageFolder(
    root=train_directory,
    transform=data_transform
)

test_dataset=datasets.ImageFolder(
    root=test_directory,
    transform=data_transform
)


class_names=train_dataset.classes
print(len(class_names))
print(len(train_dataset))
print(len(test_dataset))

BATCH_SIZE=32
train_dataLoader=DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_dataLoader=DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)
print(len(train_dataLoader))
print(len(test_dataLoader))



torch_Device="cuda" if torch.cuda.is_available() else "cpu"
print(torch_Device)

model=torchvision.models.resnet18(weights=weights).to(torch_Device)

print(model)

summary(model,input_size=(32,3,224,224))

for param in model.parameters():
    param.requires_grad = False



output_shape=len(class_names)
torch.manual_seed(42)
model.fc=nn.Sequential(
    nn.Linear( in_features=512,out_features=output_shape,bias=True)
).to(torch_Device)

summary(model,input_size=(32,3,224,224))

loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(params=model.fc.parameters(),lr=0.001)

experiment_name="fruits_experiment"
writer=SummaryWriter(log_dir=f"runs/{experiment_name}")
torch.manual_seed(42)

epochs=10

train_loss_values=[]
test_loss_values=[]
train_acc_values=[]
test_acc_values=[]


for epoch in range(epochs):
    model.train()
    train_loss=0
    train_acc=0

    for batch,(X,y) in enumerate(train_dataLoader):
        X,y=X.to(torch_Device),y.to(torch_Device)
        y_pred=model(X)
        loss=loss_function(y_pred,y)
        train_loss+=loss.item()
        y_pred_class=torch.argmax(y_pred,dim=1)
        train_acc+=(y_pred_class==y).sum().item()*100/len(y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if batch%200==0:
            print(f"Looked at :{batch}.")

    train_loss/=len(train_dataLoader)
    train_acc /= len(train_dataLoader)

    train_loss_values.append(train_loss)
    train_acc_values.append(train_acc)


    model.eval()
    test_loss=0
    test_acc=0

    with torch.inference_mode():
        for (X,y) in test_dataLoader:
            X,y=X.to(torch_Device),y.to(torch_Device)

            logits=model(X)
            loss1=loss_function(logits,y)
            test_loss+=loss1.item()

            logits_class=torch.argmax(logits,dim=1)
            test_acc+=(logits_class==y).sum().item()*100/len(logits)

        test_loss/=len(test_dataLoader)
        test_acc /= len(test_dataLoader)

        test_loss_values.append(test_loss)
        test_acc_values.append(test_acc)

        print(
        f"Train loss:{train_loss} ,Train accuracy:{train_acc},Test loss:{test_loss}, Test accuracy:{test_acc}")

        writer.add_scalars(main_tag="Loss", tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag="Accuracy", tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                           global_step=epoch)

writer.add_graph(model,input_to_model=torch.randn(32,3,224,224))
writer.close()

image_transform=transforms.Compose([
     transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])


pred_example = image_path / "seg_pred" / "seg_pred"
pred_image_ids = ("10004", "10005", "10012")
images = [pred_example / f"{image_id}.jpg" for image_id in pred_image_ids]
print(images)

model.eval()
with torch.inference_mode():
    for image_path in images:
        image=Image.open(image_path).convert("RGB")
        image=image_transform(image).unsqueeze(0).to(torch_Device)
        pred=model(image)
        pred_idx=torch.argmax(pred,dim=1)
        print(f"{image_path} -> {class_names[pred_idx]}")








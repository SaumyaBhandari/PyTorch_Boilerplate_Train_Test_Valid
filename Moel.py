import torch
import torch.optim as optim
import random
import os
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from Network import Your_Network_Name
from Dataloader import CustomDataLoader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")               #the device you want to initialize the model on, run it on cpu or gpu
# DEVICE = "cpu"
BATCH_SIZE = 32                                                                     #integer, preferable to be in a sequence of 2^n
MODEL_NAME = "Enter_The_Name_OF_Your_Model_Here"


class Model():
 
    def __init__(self, trained=False):
        self.model = YourModel().to(DEVICE)
        if trained: self.model.load_state_dict(torch.load('path_to_your_saved_model', map_location=torch.device(DEVICE)))
        self.classes = {
            0: "Class-A", 
            1: "Class-B",
            2: "Class-C",
            3: "Class-D",
            4: "Class-E",
            5: "Class-F",
            6: "Class-G",
            7: "Class-H",
            8: "Class-I",
            9: "Class-J",
        }



    def train(self, dataset, loss_func, optimizer):

        self.model.train()
        running_loss = 0.0
        running_correct = 0.0
        counter = 0

        for i, (img, label) in tqdm(enumerate(dataset), total=len(dataset)):

            counter += 1
            optimizer.zero_grad()
            image, label = img.to(DEVICE), label.to(DEVICE)
            output = self.model(image)
            loss = loss_func(output, label)
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            # calculate accuracy
            pred = output.argmax(1)
            correct = pred == label
            running_correct += correct.sum().item()

        # loss and accuracy for a complete epoch
        epoch_loss = running_loss / (counter*BATCH_SIZE)
        epoch_acc = 100. * (running_correct / (counter*BATCH_SIZE))

        return epoch_loss, epoch_acc



    def validate(self, dataset):

        self.model.eval()
        running_correct = 0.0
        counter = 0

        with torch.no_grad():
            for i, (img, label) in tqdm(enumerate(dataset), total=len(dataset)):
                counter += 1
                img, label = img.to(DEVICE), label.to(DEVICE)
                outputs = self.model(img)

                #calculate accuracy
                pred = outputs.argmax(1)
                correct = pred == label
                running_correct += correct.sum().item()

        # Accuracy for a complete epoch
        epoch_acc = 100. * (running_correct / (counter*BATCH_SIZE))
        return epoch_acc



    def test(self, dataset, epoch):

        running_correct = 0.0
        counter = 0

        num = random.randint(0, len(dataset)-1)
        self.model.eval()
        with torch.no_grad():
            for i, (img, label) in tqdm(enumerate(dataset), total=len(dataset)):
                counter += 1
                img, label = img.to(DEVICE), label.to(DEVICE)
                outputs = self.model(img)


                #calculate accuracy
                pred = outputs.argmax(1)
                correct = pred == label
                running_correct += correct.sum().item()
                
                #save a random testing sample
                if i == num:
                    try:
                        os.makedirs(f"saved_samples/{MODEL_NAME}", exist_ok=True)
                    except:
                        pass
                    sample = random.randint(0, BATCH_SIZE//2)
                    image = img[sample, :, :, :].cpu().numpy().transpose((1, 2, 0))
                    image = (image * 255).astype('uint8')
                    image = Image.fromarray(image)
                    draw = ImageDraw.Draw(image)
                    real_label = self.classes[label[sample].item()]
                    pred_label = self.classes[pred[sample].item()]
                    draw.text((image.width - 200, 0), f"Real: {real_label}", fill='red')
                    draw.text((image.width - 200, 20), f"Predicted: {pred_label}", fill='blue')
                    image.save(f"saved_samples/{MODEL_NAME}/{num}.jpg")

        # loss and accuracy for a complete epoch
        epoch_acc = 100. * (running_correct / (counter))
    
        return epoch_acc



 
    def fit(self, epochs, lr):

        print(f"Using {DEVICE} device...")
        print("Loading Datasets...")
        train_data, val_data, test_data = CustomDataLoader(BATCH_SIZE).get_data()
        print("Dataset Loaded.")

        print("Initializing Parameters...")
        total_params = sum(p.numel() for p in self.model.parameters())
        print("Total parameters of the model is: {:.2f}{}".format(total_params / 10**(3 * min(len(str(total_params)) // 3, len(["", "K", "M", "B", "T"]) - 1)), ["", "K", "M", "B", "T"][min(len(str(total_params)) // 3, len(["", "K", "M", "B", "T"]) - 1)]))

        print(f"Initializing the Optimizer")
        optimizer = optim.AdamW(self.model.parameters(), lr)
        print(f"Beginning to train...")

        crossEntropyLoss = nn.CrossEntropyLoss()
        val_acc_epochs = []
        writer = SummaryWriter(f'runs/{MODEL_NAME}/')

        os.makedirs("checkpoints/", exist_ok=True)
        os.makedirs("saved_model/", exist_ok=True)


        for epoch in range(1, epochs+1):

            print(f"Epoch No: {epoch}")

            #training loop
            train_loss, train_acc = self.train(dataset=train_data, loss_func=crossEntropyLoss, optimizer=optimizer)

            #validation loop and save the model with best validation accuracy
            val_acc = self.validate(dataset=val_data)
            val_acc_epochs.append(val_acc)
            if max(val_acc_epochs) == val_acc:
                torch.save(self.model.state_dict(), f"checkpoints/{MODEL_NAME}.pth")

            print(f"Train Loss:{train_loss}, Train Accuracy:{train_psnr}, Validation Accuracy:{val_psnr}")

            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Acc/Train", train_acc, epoch)
            writer.add_scalar("Acc/Val", val_acc, epoch)

            if epoch%5==0:
                print("Testing...")
                test_acc = self.test(test_data, epoch)
                print(f"Test Accuracy: {test_acc}")
                writer.add_scalar("Acc/Test", test_acc)
                print("Saving model")
                torch.save(self.model.state_dict(), f"saved_model/{MODEL_NAME}_{epoch}.pth")
                print("Model Saved")

    
            print("Epoch Completed. Proceeding to next epoch...")

        print(f"Training Completed for {epochs} epochs.")




    def infer_a_random_sample(self):
        
        try:
            os.makedirs(f"test_samples/{MODEL_NAME}", exist_ok=True)
        except:
            pass
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        with open('Path_to_test_csv_file', newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            rows = list(csvreader)
            random_row = random.choice(rows)
            path = random_row[0]
            label = random_row[1]

            image = Image.open(path)
            imageT = transform(image).unsqueeze(0).to(DEVICE)
            outputs = self.model(imageT)
            pred = outputs.argmax(1)
            pred_label = self.classes[pred.item()]
            print(pred_label)
            print(label)

            draw = ImageDraw.Draw(image)
            draw.text((image.width - 200, 0), f"Real: {label}", fill='red')
            draw.text((image.width - 200, 20), f"Predicted: {pred_label}", fill='blue')
            image.save(f"test_samples/{MODEL_NAME}/{label} -> {pred_label}.jpg")
            print("Saved a sample")



        def infer_a_sample(self, image):

            image = image.to(DEVICE)
            self.model.eval()
            # Forward pass the image through the model.
            prediction = nn.Softmax(dim=1)(self.model(image)).max(1)
            class_prob, class_index = round(prediction.values.item(), 3), prediction.indices.item()
            class_name = self.classes[class_index]
            return f'{class_name}: {class_prob*100}%'



model = Model()
model.fit(250, 5e-5)
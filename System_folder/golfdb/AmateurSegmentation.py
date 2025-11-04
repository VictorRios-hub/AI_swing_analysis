import os 
import argparse
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
from eval import ToTensor, Normalize
from model import EventDetector



class SampleVideo(Dataset):
    def __init__(self, path, input_size=160, transform=None):
        self.path = path
        self.input_size = input_size
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        cap = cv2.VideoCapture(self.path)
        frame_size = [cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)]
        ratio = self.input_size / max(frame_size)
        new_size = tuple([int(x * ratio) for x in frame_size])
        delta_w = self.input_size - new_size[1]
        delta_h = self.input_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # preprocess and return frames
        images = []
        for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            _, img = cap.read()
            resized = cv2.resize(img, (new_size[1], new_size[0]))
            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=[0.406 * 255, 0.456 * 255, 0.485 * 255])  # ImageNet means (BGR)

            b_img_rgb = cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB)
            images.append(b_img_rgb)
        cap.release()
        labels = np.zeros(len(images)) # only for compatibility with transforms
        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


def process_segmentation():
    
#This system segments each amateur moment of swing into seperate images and stores the images in their respective files
    event_names = {
        0: 'Address',
        1: 'Toe-up',
        2: 'Mid-backswing (arm parallel)',
        3: 'Top',
        4: 'Mid-downswing (arm parallel)',
        5: 'Impact',
        6: 'Mid-follow-through (shaft parallel)',
        7: 'Finish'
    }
    # Get the directory of the current script
    current_script_directory = os.path.dirname(os.path.abspath(__file__))
    relative_directory = 'VideoData/Amateur-vids'
    
    # Construct the absolute path
    directory = os.path.join(current_script_directory, relative_directory)
    
    fileitr = 0
    for filename in os.listdir(directory):
        fileitr += 1
        f = os.path.join(directory, filename)        
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--path', help='Path to video that you want to test', default= f)
        parser.add_argument('-s', '--seq-length', type=int, help='Number of frames to use per forward pass', default=48)
        args = parser.parse_args()
        seq_length = args.seq_length

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)

        # print('Preparing video: {}'.format(args.path))

        ds = SampleVideo(args.path, transform=transforms.Compose([ToTensor(),
                                                                Normalize([0.485, 0.456, 0.406],
                                                                            [0.229, 0.224, 0.225])]))

        dl = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

        model = EventDetector(pretrain=True,
                            width_mult=1.,
                            lstm_layers=1,
                            lstm_hidden=256,
                            bidirectional=True,
                            dropout=False)

        try:
            # Get the directory of the current script
            current_script_directory = os.path.dirname(os.path.abspath(__file__))
            relative_directory = 'models/swingnet_1800.pth.tar'
            
            # Construct the absolute path
            model_dir = os.path.join(current_script_directory, relative_directory)
    
            save_dict = torch.load(model_dir,map_location=device)
        except:
                print("Model weights not found. Download model weights and place in 'models' folder. See README for instructions")

        model.load_state_dict(save_dict['model_state_dict'])
        model.to(device)
        model.eval()
        print("Loaded model weights")

        print('Testing...')
        for sample in dl:
            images = sample['images']
            # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
            batch = 0
            while batch * seq_length < images.shape[1]:
                if (batch + 1) * seq_length > images.shape[1]:
                    image_batch = images[:, batch * seq_length:, :, :, :]
                else:
                    image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
                    
                    if device == torch.device('cuda') : 
                        logits = model(image_batch.cuda())
                    else : 
                        logits = model(image_batch.cpu())

                if batch == 0:
                    probs = F.softmax(logits.data, dim=1).cpu().numpy()
                else:
                    probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
                batch += 1

        events = np.argmax(probs, axis=0)[:-1]
        print('Predicted event frames: {}'.format(events))
        cap = cv2.VideoCapture(args.path)
    
    
        write_images = True
        confidence = []
        for i, e in enumerate(events):
            confidence.append(probs[e, i]) 
            print('confidence: {}'.format([round(c, 3) for c in confidence]))
            
        
        label = ['Address', 'Toe-up', 'Mid-Backswing (arm parallel)', 'Top', 'Mid-Downswing (arm parallel)', 'Impact', 'Mid-Follow-Through (shaft parallel)', 'Finish']
        
        paths = []
        #write data to file
        for i, e in enumerate(events):
            cap.set(cv2.CAP_PROP_POS_FRAMES, e)
            _, img = cap.read()
            # cv2.putText(img, '{:.3f}'.format(confidence[i]), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 255))
            #cv2.imshow(event_names[i], img)  
            
            # Get the directory of the current script
            current_script_directory = os.path.dirname(os.path.abspath(__file__))
            relative_directory = 'models/swingnet_1800.pth.tar'
            
            # Construct the absolute path
            model_dir = os.path.join(current_script_directory, relative_directory)
            
            # Get the directory of the current script
            current_script_directory = os.path.dirname(os.path.abspath(__file__))
            relative_directory = 'ImageData/Am_Swing_events/'
            
            # Construct the absolute path
            out_directory = os.path.join(current_script_directory, relative_directory)
    
            output_directory = out_directory + label[i] + "/"
            os.makedirs(output_directory, exist_ok=True)
            output_file = output_directory + event_names[i] + str(fileitr) + ".jpg"
            cv2.imwrite(output_file, img)  
            paths.append(output_file)
            #writeToStr = "data/amateur_data/"  + label[i] + ' Confidence: {}'.format([np.round(c, 3) for c in confidence]) + "/" + event_names[i] +""+ str(fileitr) + ".jpg", img
            #print('sending to : '+ writeToStr)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return paths
        
if __name__ == '__main__':
    paths = process_segmentation()
    # print(paths)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "ename": "BrokenPipeError",
     "evalue": "[Errno 32] Broken pipe",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mBrokenPipeError\u001b[0m                           Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-7-0176baf7f7e5>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     43\u001b[0m     \u001b[0mvalloader\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtorch\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mutils\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdata\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mDataLoader\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mvalset\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mbatch_size\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m4\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mshuffle\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mFalse\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnum_workers\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     44\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 45\u001b[1;33m     \u001b[0mdataiter\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0miter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtrainloader\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     46\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     47\u001b[0m     \u001b[0mdataiter\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mnet\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\envs\\py4dp\\lib\\site-packages\\torch\\utils\\data\\dataloader.py\u001b[0m in \u001b[0;36m__iter__\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    276\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0m_SingleProcessDataLoaderIter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    277\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 278\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0m_MultiProcessingDataLoaderIter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    279\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    280\u001b[0m     \u001b[1;33m@\u001b[0m\u001b[0mproperty\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\envs\\py4dp\\lib\\site-packages\\torch\\utils\\data\\dataloader.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, loader)\u001b[0m\n\u001b[0;32m    680\u001b[0m             \u001b[1;31m#     before it starts, and __del__ tries to join but will get:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    681\u001b[0m             \u001b[1;31m#     AssertionError: can only join a started process.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 682\u001b[1;33m             \u001b[0mw\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mstart\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    683\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_index_queues\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mindex_queue\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    684\u001b[0m             \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_workers\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mw\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\envs\\py4dp\\lib\\multiprocessing\\process.py\u001b[0m in \u001b[0;36mstart\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    103\u001b[0m                \u001b[1;34m'daemonic processes are not allowed to have children'\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    104\u001b[0m         \u001b[0m_cleanup\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 105\u001b[1;33m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_popen\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_Popen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    106\u001b[0m         \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_sentinel\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_popen\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msentinel\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    107\u001b[0m         \u001b[1;31m# Avoid a refcycle if the target function holds an indirect\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\envs\\py4dp\\lib\\multiprocessing\\context.py\u001b[0m in \u001b[0;36m_Popen\u001b[1;34m(process_obj)\u001b[0m\n\u001b[0;32m    221\u001b[0m     \u001b[1;33m@\u001b[0m\u001b[0mstaticmethod\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    222\u001b[0m     \u001b[1;32mdef\u001b[0m \u001b[0m_Popen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mprocess_obj\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 223\u001b[1;33m         \u001b[1;32mreturn\u001b[0m \u001b[0m_default_context\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_context\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mProcess\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_Popen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mprocess_obj\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    224\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    225\u001b[0m \u001b[1;32mclass\u001b[0m \u001b[0mDefaultContext\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mBaseContext\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\envs\\py4dp\\lib\\multiprocessing\\context.py\u001b[0m in \u001b[0;36m_Popen\u001b[1;34m(process_obj)\u001b[0m\n\u001b[0;32m    320\u001b[0m         \u001b[1;32mdef\u001b[0m \u001b[0m_Popen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mprocess_obj\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    321\u001b[0m             \u001b[1;32mfrom\u001b[0m \u001b[1;33m.\u001b[0m\u001b[0mpopen_spawn_win32\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mPopen\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 322\u001b[1;33m             \u001b[1;32mreturn\u001b[0m \u001b[0mPopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mprocess_obj\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    323\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    324\u001b[0m     \u001b[1;32mclass\u001b[0m \u001b[0mSpawnContext\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mBaseContext\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\envs\\py4dp\\lib\\multiprocessing\\popen_spawn_win32.py\u001b[0m in \u001b[0;36m__init__\u001b[1;34m(self, process_obj)\u001b[0m\n\u001b[0;32m     63\u001b[0m             \u001b[1;32mtry\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     64\u001b[0m                 \u001b[0mreduction\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdump\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mprep_data\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mto_child\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 65\u001b[1;33m                 \u001b[0mreduction\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdump\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mprocess_obj\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mto_child\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     66\u001b[0m             \u001b[1;32mfinally\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     67\u001b[0m                 \u001b[0mset_spawning_popen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\envs\\py4dp\\lib\\multiprocessing\\reduction.py\u001b[0m in \u001b[0;36mdump\u001b[1;34m(obj, file, protocol)\u001b[0m\n\u001b[0;32m     58\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0mdump\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mfile\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mprotocol\u001b[0m\u001b[1;33m=\u001b[0m\u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     59\u001b[0m     \u001b[1;34m'''Replacement for pickle.dump() using ForkingPickler.'''\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 60\u001b[1;33m     \u001b[0mForkingPickler\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfile\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mprotocol\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mdump\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mobj\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     61\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     62\u001b[0m \u001b[1;31m#\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mBrokenPipeError\u001b[0m: [Errno 32] Broken pipe"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import glob\n",
    "import os\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torchvision\n",
    "import torchvision.transforms as transforms\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "\n",
    "import cv2\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "INPUT_DIR = 'data/VOCdevkit/VOC2007/JPEGImages/'\n",
    "\n",
    "import test\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    \n",
    "    filelist = glob.glob(INPUT_DIR + '*.jpg')\n",
    "    \n",
    "    transform = transforms.ToTensor()\n",
    "\n",
    "    dataset = ImageDataset(filelist, transform)\n",
    "    \n",
    "    train_indices = list(range(1000,len(dataset)))\n",
    "    validation_indices = list(range(1000))\n",
    "\n",
    "    trainset = torch.utils.data.Subset(dataset, train_indices)\n",
    "\n",
    "    train_subset = torch.utils.data.Subset(trainset, list(range(100)))\n",
    "\n",
    "    valset = torch.utils.data.Subset(dataset, validation_indices)\n",
    "    \n",
    "    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=1)\n",
    "\n",
    "    train_subloader = torch.utils.data.DataLoader(train_subset, batch_size=4, shuffle=True, num_workers=1)\n",
    "\n",
    "    valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False, num_workers=1)\n",
    "    \n",
    "    dataiter = iter(trainloader)\n",
    "    \n",
    "    dataiter.net()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "ename": "ImportError",
     "evalue": "cannot import name 'ImageDataset'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mImportError\u001b[0m                               Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-13-d8e73c12f523>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[0mtest\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mImageDataset\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;31mImportError\u001b[0m: cannot import name 'ImageDataset'"
     ]
    }
   ],
   "source": [
    "from test import ImageDataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "test"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

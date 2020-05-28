import os
import sys
import math
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.autograd import Variable
from KittiDataset import KittiDataset


def obstacleLossFun(outputBatch, obstacleBatch):
    outFlat = outputBatch.view(-1)
    inpFlat = obstacleBatch.view(-1)
    intersection = (outFlat * inpFlat).abs().sum()
    return intersection / len(outFlat)

def heatmapAccuracy(outputMap, labelMap, thr=1.5):
    pred = np.unravel_index(outputMap.argmax(), outputMap.shape)
    gt = np.unravel_index(labelMap.argmax(), labelMap.shape)

    dist = math.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)
    if dist <= thr:
        return 1, dist, (pred[0], pred[1])
    return 0, dist, (pred[0], pred[1])

def weightedMSE(outputMap, labelMap, weightMap):
    out = (outputMap - labelMap) ** 2
    out = out * weightMap
    loss = out.sum(0)
    return loss

def weightMatrix(labelMap):
    labelClone = labelMap.clone()
    weightMat = labelMap.clone()
    num_nonzeros = torch.nonzero(labelClone).size(0)
    num_zeros = cmd.imageHeight * cmd.imageWidth - num_nonzeros
    weightMat[labelClone == 0] = float(1) / num_zeros
    weightMat[labelClone != 0] = float(1) / num_nonzeros
    return weightMat

# Returns True if the model is LSTM based
def loadModel(modelType, imageWidth, imageHeight, activation, initType, numChannels, batchnorm, dilation,
              hiddenUnits=512, fcSize=4096, softmax=False):
    # Encoder Decoder CNN without LSTM / RNN units
    if modelType == "edCNN_wp":
        from Model import EnDeWithPooling
        model = EnDeWithPooling(activation, initType, numChannels, batchnorm, softmax)
        model.init_weights()
        return model, False

    if modelType == "convLSTM":
        from Model import EnDeConvLSTM
        model = EnDeConvLSTM(activation, initType, numChannels, imageHeight, imageWidth, batchnorm=batchnorm,
                             softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "convLSTM_ws":
        from Model import EnDeConvLSTM_ws
        model = EnDeConvLSTM_ws(activation, initType, numChannels, imageHeight, imageWidth, batchnorm=batchnorm,
                                softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "skipLSTM":
        from Model import SkipLSTMEnDe
        model = SkipLSTMEnDe(activation, initType, numChannels, imageHeight, imageWidth, batchnorm=batchnorm,
                             softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "enDeLayerNorm":
        from Model import EnDeLayerNorm_ws
        model = EnDeLayerNorm_ws(activation, initType, numChannels, imageHeight, imageWidth, softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "enDeLayerNorm1D":
        from Model import EnDeLayerNorm1D_ws
        model = EnDeLayerNorm1D_ws(activation, initType, numChannels, imageHeight, imageWidth, softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "skipLayerNorm":
        from Model import SkipLSTMLayerNorm
        model = SkipLSTMLayerNorm(activation, initType, numChannels, imageHeight, imageWidth, softmax=softmax)
        model.init_weights()
        return model, True

    if modelType == "skipLayerNorm1D":
        from Model import SkipLSTMLayerNorm1D
        model = SkipLSTMLayerNorm1D(activation, initType, numChannels, imageHeight, imageWidth, softmax=softmax)
        model.init_weights()
        return model, True
args = {}
# MODEL OPTIONS
args['initType'] = 'default'
args['activation'] = 'relu'
args['imageWidth'] = 256
args['imageHeight'] = 256
args['modelType'] = "skipLSTM"
args['dilation'] = True # for convolution
args['lossOT'] = True # safety loss
args['usePrev'] = False # Use previous prediction to predict next, in LSTM
args['lane'] = True
args['obstacles'] = True
args['road'] = True
args['vehicles'] = True
args['list'] = 4
args['resume_path'] = '/datasets/home/44/344/abkandoi/trajectory_prediction_INFER/ablation_cache/skipLSTM/split-0'
args['resume'] = False  #bool


# HYPERPARAMETERS
args['lr'] = 0.000100
args['momentum'] = 0.900000
args['weightDecay'] = 0.0
args['lrDecay'] = 0.0
args['nepochs'] = 60
args['beta1'] = 0.90000
args['beta2'] = 0.999
args['gradClip'] = 10.0
args['optMethod'] = 'adam'
args['batchnorm'] = False
args['seqLen'] = 1
args['lossFun'] = 'default'
args['scaleFactor'] = False
args['softmax'] = False
args['gamma'] = 0.0
args['futureEpochs'] = 10.0
args['futureFrames'] = 20.0
args['scheduledSampling'] = False
args['minMaxNorm'] = False
args['lambda1'] = 1.0


# DATASET
args['dataDir'] = '/datasets/home/44/344/abkandoi/trajectory_prediction_INFER/INFER-datasets/kitti'
args['augmentation'] = False
args['augmentationProb'] = 0.3
args['groundTruth'] = True
args['csvDir'] = '/datasets/home/44/344/abkandoi/trajectory_prediction_INFER/INFER-datasets/kitti/final-validation'
args['trainPath'] = 'train0.csv'
args['valPath'] = 'test0.csv'
args['expID'] = 'split-0'

channels = []
if args['lane'] != "False":
    channels.append("lane")
if args['obstacles'] != "False":
    channels.append("obstacles")
if args['road'] != "False":
    channels.append("road")
if args['vehicles'] != "False":
    channels.append("vehicles")
args['channels'] = channels

print("Channels Used: ", channels)

model, isLSTM = loadModel(args['modelType'], args['imageWidth'], args['imageHeight'],
                          args['activation'], args['initType'], len(channels) + 1,
                          args['batchnorm'], args['dilation'])

if args['resume']:
    print('TRAINING TO BE RESUMED!')
    print('Loading checkpoint.tar')
    checkpoint = torch.load(os.path.join(args['resume_path'], 'checkpoint.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])
    lastEpoch = checkpoint['epoch']
    print('lastEpoch is {}'.format(lastEpoch))
    print('will resume training from epoch {}'.format(lastEpoch+1))

model = model.cuda()
# Make Directory Structure to Save the Models:
baseDir = '/datasets/home/44/344/abkandoi/trajectory_prediction_INFER'
print('baseDir is {}'.format(baseDir))
expDir = os.path.join(baseDir, 'ablation_cache', args['modelType'], args['expID'])
print('expDir is {}'.format(expDir))
lossDir = os.path.join(expDir, 'loss')
os.makedirs(expDir, exist_ok=True)
os.makedirs(lossDir, exist_ok=True)

# Save the command line arguments
with open(os.path.join(expDir, 'args.txt'), 'w') as cmdFile:
    for key in args:
        cmdFile.write(key + ' ' + str(args[key]) + '\n')


# Loss Function
criterion = nn.MSELoss()

# Optimizer
optimizer = None
if args['optMethod'] == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(args['beta1'], args['beta2']), weight_decay=args['weightDecay'])
elif args['optMethod'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'], weight_decay=args['weightDecay'],
                          nesterov=False)
elif args['optMethod'] == 'amsgrad':
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], betas=(args['beta1'], args['beta2']), weight_decay=args['weightDecay'],
                           amsgrad=True)


if args['resume']:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('Loaded optimizer state dictionary')

# Default CUDA tensor
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Scale factor to scale the label channel
scf = 1
if args['scaleFactor']:
    scf = args['imageHeight'] * args['imageWidth']

if args['csvDir'] is None:
    args['csvDir'] = args['dataDir']

print("-" * 100)
print("scf is {}".format(scf))
print("Loss: ", args['lossFun'])
print("Data Dir: ", args['dataDir'])
print("CSV Dir: ", args['csvDir'])

trainInfoPath = os.path.join(args['csvDir'], args['trainPath'])
trainDataset = KittiDataset(args['dataDir'], height=args['imageHeight'], width=args['imageWidth'], train=True,
                            infoPath=trainInfoPath, augmentation=args['augmentation'],
                            augmentationProb=args['augmentationProb'], channels=args['channels'],
                            groundTruth=args['groundTruth'])

valInfoPath = os.path.join(args['csvDir'], args['valPath'])
valDataset = KittiDataset(args['dataDir'], height=args['imageHeight'], width=args['imageWidth'], train=False,
                          infoPath=valInfoPath, channels=args['channels'], groundTruth=args['groundTruth'])

epochTrainLoss = []
epochValidLoss = []

if args['resume']:
    epochTrainLoss = checkpoint['train_loss']
    epochValidLoss = checkpoint['valid_loss']
    print('Loaded epochTrainLoss and epochValidLoss')


# Saving Model Weights
best_model_weights = copy.deepcopy(model.state_dict())
best_loss = 100000000

# Saving Future Model Weights
best_model_weights_future = copy.deepcopy(model.state_dict())
best_loss_future = 100000000

if args['resume']:
    print('Loading checkpoint_best.tar')
    checkpoint_best = torch.load(os.path.join(args['resume_path'], 'checkpoint_best.tar'))
    best_loss = checkpoint_best['valid_loss'][-1]
    # update best_loss_future if checkpoint for it exists
    if lastEpoch > int(args['futureEpochs']):
        print('Loading checkpoint_future_best.tar')
        checkpoint_future_best = torch.load(os.path.join(args['resume_path'], 'checkpoint_future_best.tar'))
        best_loss_future = checkpoint_future_best['valid_loss'][-1]


# Loss History
lossHistory = []

epochRange = range(args['nepochs'])
if args['resume']:
    epochRange = range(lastEpoch + 1, args['nepochs'])

for epoch in epochRange:
    print("-" * 100)
    print("Epoch No: {}".format(epoch))
    startTime = time.time()
    model.train()
    # Total Loss of one trajectory
    loss = None
    # Hidden states of the LSTM
    state = None
    # Network Prediction
    out = None
    prevOut = None
    # Number of samples forwarded
    count = 0
    # Apply Loss after every batch only
    labelBatch = None
    outputBatch = None
    obstacleBatch = None
    weightBatch = None

    # Updated train and val loss after each epoch
    trainLossPerEpoch = []
    validLossPerEpoch = []

    seqLoss = []
    curSeqNum = 0

    # Training Loop
    for i in range(len(trainDataset)):
        if loss is None:
            # First pair to be forwarded, hence zero grad
            model.zero_grad()

        grid, kittiSeqNum, vehicleId, frame1, frame2, endOfSequence, offset, numFrames, augmentation = trainDataset[i]

        # The Last Channel is the target frame and first n - 1 are source frames
        inp = grid[:-1, :].unsqueeze(0).cuda()
        currLabel = grid[-1:, :].unsqueeze(0).cuda()
        # weightMat = weightMatrix(currLabel)
        currOutput = None
        obstacle = None

        if labelBatch is None:
            labelBatch = scf * grid[-1:, :].unsqueeze(0).cuda()
        else:
            labelBatch = torch.cat((labelBatch, (scf * (grid[-1:, :])).unsqueeze(0).cuda()), 0)
        
        if obstacleBatch is None:
            obstacleBatch = grid[2, :].unsqueeze(0).cuda()
        else:
            obstacleBatch = torch.cat((obstacleBatch, grid[2, :].unsqueeze(0).cuda()), 0)

        # Pass the future predictions after pre-conditioning the LSTM
        if offset >= int(args['futureFrames']) and epoch > int(args['futureEpochs']):
            new_inp = inp.clone().squeeze(0)
            if args['minMaxNorm']:
                mn, mx = torch.min(prevOut), torch.max(prevOut)
                prevOut = (prevOut - mn) / (mx - mn)
            new_inp[0] = prevOut
            inp = new_inp.unsqueeze(0).cuda()

        if isLSTM:
            if args['modelType'] in ["skipLSTM", "skipLayerNorm1D", "skipLayerNorm"]:
                # 3 LSTMs => 3 hidden states
                out = model.forward(inp, state)
                currOutputMap = out.clone()
                state = (model.h, model.c, model.h1, model.c1, model.h2, model.c2)
            else:
                # Simple LSTM => Only 1 hidden state
                out = model.forward(inp, state)
                currOutputMap = out.clone()
                state = (model.h, model.c)
        else:
            # No LSTM => No hidden state
            # Forward the input and obtain the result
            out = model.forward(inp)
            currOutputMap = out.clone()

        if outputBatch is None:
            outputBatch = out
        else:
            outputBatch = torch.cat((outputBatch, out), 0)

        count += 1
        prevOut = currOutputMap.detach().cpu().squeeze(0).squeeze(0)
        currOutputMap = currOutputMap.detach().cpu().numpy().squeeze(0).squeeze(0)
        currLabel = currLabel.detach().cpu().numpy().squeeze(0).squeeze(0)
        _, dist, predCoordinates = heatmapAccuracy(currOutputMap, currLabel)

        if offset >= int(args['futureFrames']):
            seqLoss.append(dist)

        if count == args['seqLen'] or endOfSequence is True:
            # Regularization
            l2_reg = None
            for W in model.parameters():
                if l2_reg is None:
                    l2_reg = W.norm(2)
                else:
                    l2_reg = l2_reg + W.norm(2)

            l2_reg = args['gamma'] * l2_reg

            if args['lossFun'] == "default":
                if args['lossOT']:
                    obstacleLoss = obstacleLossFun(outputBatch, obstacleBatch)
                else:
                    obstacleLoss = 0.0
                mseloss = criterion(outputBatch, labelBatch)
                loss = mseloss + args['lambda1'] * obstacleLoss
                #print('MSELoss is {}'.format(mseloss))
                #print('lambda1 * obstacle loss is {}'.format(args['lambda1'] * obstacleLoss))
                # loss = sum([criterion(outputBatch, labelBatch), l2_reg, obstacleLoss])
            elif args['lossFun'] == "weightedMSE":
                if args['lossOT']:
                    obstacleLoss = obstacleLossFun(outputBatch, obstacleBatch)
                else:
                    obstacleLoss = 0.0
                loss = weightedMSE(outputBatch, labelBatch, weightBatch) + args['lambda1'] * obstacleLoss
                # loss = sum([weightedMSE(outputBatch, labelBatch, weightBatch), l2_reg, obstacleLoss])

            if isLSTM:
                if endOfSequence is True:
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
            else:
                loss.backward()

            if args['gradClip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['gradClip'])

            optimizer.step()

            # Reset
            loss = None
            labelBatch = None
            obstacleBatch = None
            outputBatch = None
            count = 0

            if endOfSequence is True:
                if numFrames >= 60:
                    trainLossPerEpoch.append(np.mean(seqLoss))
                    print("kittiSeq: {}, vehicleId: {}, trainSeqNo: {}, numFrames: {}, Augmentation: {}, Seq Loss: {}".format(
                        kittiSeqNum, vehicleId, curSeqNum, numFrames, augmentation, np.mean(seqLoss)))
                    lossHistory.append(["Training", kittiSeqNum, vehicleId, curSeqNum, augmentation, np.mean(seqLoss)])

                curSeqNum += 1
                if isLSTM:
                    state = None
                seqLoss = []

    print("Average train loss: ", np.mean(trainLossPerEpoch))
    print("For training : --- %s seconds ---" % (time.time() - startTime))

    if np.mean(trainLossPerEpoch) >= 15:
        print("Params Value")
        for name, param in model.named_parameters():
            print("Name: ", name)
            print("Grad: ", param.grad.data.norm(2.))

    epochTrainLoss.append(np.mean(trainLossPerEpoch))

    # Validation
    startTime = time.time()
    state = None
    model.eval()
    seqLoss = []
    curSeqNum = 0
    for i in range(len(valDataset)):
        grid, kittiSeqNum, vehicleId, frame1, frame2, endOfSequence, offset, numFrames, augmentation = valDataset[i]

        # The Last Channel is the target frame and first n - 1 are source frames
        inp = grid[:-1, :].unsqueeze(0).cuda()
        label = grid[-1:, :].unsqueeze(0).cuda()

        if offset >= int(args['futureFrames']) and epoch > int(args['futureEpochs']):
            new_inp = inp.clone()
            new_inp = new_inp.squeeze(0)
            if args['minMaxNorm']:
                mn, mx = torch.min(prevOut), torch.max(prevOut)
                prevOut = (prevOut - mn) / (mx - mn)
            new_inp[0] = prevOut
            inp = new_inp.unsqueeze(0).cuda()

        if isLSTM:
            if args['modelType'] in ["skipLSTM", "skipLayerNorm1D", "skipLayerNorm"]:
                out = model.forward(inp, state)
                currOutputMap = out.clone()
                state = (model.h, model.c, model.h1, model.c1, model.h2, model.c2)
            else:
                out = model.forward(inp, state)
                currOutputMap = out.clone()
                state = (model.h, model.c)
        else:
            out = model.forward(inp)
            currOutputMap = out.clone()

        prevOut = currOutputMap.detach().cpu().squeeze(0).squeeze(0)
        outputMap = out.detach().cpu().numpy().squeeze(0).squeeze(0)
        labelMap = label.detach().cpu().numpy().squeeze(0).squeeze(0)
        _, dist, predCoordinates = heatmapAccuracy(outputMap, labelMap)

        if offset >= int(args['futureFrames']):
            seqLoss.append(dist)

        if endOfSequence:
            state = None
            if offset >= int(args['futureFrames']):
                if numFrames >= 60:
                    validLossPerEpoch.append(np.mean(seqLoss))
                    print("kittiSeq: {}, vehicleId: {}, valSeqNo: {}, numFrames: {}, Augmentation: {}, Seq Loss: {}".format(
                        kittiSeqNum, vehicleId, curSeqNum, numFrames, augmentation, np.mean(seqLoss)))
                    lossHistory.append(["Validation", kittiSeqNum, vehicleId, curSeqNum, augmentation, np.mean(seqLoss)])
            seqLoss = []
            curSeqNum += 1

    avgValidLoss = np.mean(validLossPerEpoch)
    print("Average valid loss: ", avgValidLoss)
    epochValidLoss.append(avgValidLoss)

    # best model checkpoint
    if avgValidLoss < best_loss:
        best_loss = avgValidLoss
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": epochTrainLoss,
            "valid_loss": epochValidLoss
        }
        torch.save(checkpoint, os.path.join(expDir, 'checkpoint_best.tar'))
        torch.save(model, os.path.join(expDir, 'model_best.pth'))

    # checkpoint every epoch
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": epochTrainLoss,
        "valid_loss": epochValidLoss
    }
    torch.save(checkpoint, os.path.join(expDir, 'checkpoint.tar'))
    torch.save(model, os.path.join(expDir, 'model.pth'))

    print('CHECKPOINT: saved to {}'.format(os.path.join(expDir, 'checkpoint.tar')))

    if epoch > int(args['futureEpochs']):
        checkpoint_future = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": epochTrainLoss,
            "valid_loss": epochValidLoss
        }
        torch.save(checkpoint_future, os.path.join(expDir, 'checkpoint_future.tar'))
        torch.save(model, os.path.join(expDir, 'model_future.pth'))

        if avgValidLoss < best_loss_future:
            best_loss_future = avgValidLoss
            best_model_weights_future = copy.deepcopy(model.state_dict())
            checkpoint_future = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epochTrainLoss,
                "valid_loss": epochValidLoss
            }
            torch.save(checkpoint_future, os.path.join(expDir, 'checkpoint_future_best.tar'))
            torch.save(model, os.path.join(expDir, 'model_future_best.pth'))

    print("For Validation: --- %s seconds ---" % (time.time() - startTime))

    if epoch % 5 == 0:
        fig, ax = plt.subplots(1)
        ax.plot(range(len(epochTrainLoss)), epochTrainLoss, 'r', label='Train Loss')
        ax.plot(range(len(epochValidLoss)), epochValidLoss, 'g', label='Valid Loss')
        ax.legend()
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        fig.savefig(os.path.join(expDir, 'loss', 'loss_epoch'))
        plt.close()

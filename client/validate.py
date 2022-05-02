import sys
from read_data import read_data
import json
import yaml
import torch
import collections
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np


def np_to_weights(weights_np):
    weights = collections.OrderedDict()
    for w in weights_np:
        weights[w] = torch.tensor(weights_np[w])
    return weights

def validate(model, data, settings):
    print("-- RUNNING VALIDATION --", flush=True)
    # The data, split between train and test sets. We are caching the partition in 
    # the container home dir so that the same data subset is used for 
    # each iteration.

    def r2_loss(output, target):
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - output) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2

    def evaluate(model, loss, dataloader):
        model.eval()
        train_loss = 0
        train_loss1 = 0
        train_correct = 0
        with torch.no_grad():
            for x, y in dataloader:
    
                batch_size = x.shape[0]
                x = torch.squeeze(x, 1)
                x1=x.float().numpy()
                x_float = torch.from_numpy(x1)

                output = model.forward(x_float)

                print('###################################################################################')
                print(type(output))
                print(type(y))
                print('###################################################################################')
                print(output)
                print('###################################################################################')
                
                input = torch.zeros((batch_size, 128), dtype=torch.float32)
                input_mask = torch.zeros((batch_size, 128), dtype=torch.int32)
                for i, row in enumerate(x1):
                    input_mask[i, int(torch.FloatTensor(row)[70401].item())] = 1
                    input[i, int(torch.FloatTensor(row)[70401].item())] = float(y[i].item())

                train_loss += batch_size * loss(output, input, input_mask).item()


                # pred = output.argmax(dim=1, keepdim=True)
                # train_correct += pred.eq(y.view_as(pred)).sum().item()


                output1 = torch.squeeze(output['shift_mu'], 2)
                # r2 = r2_loss(output1, y)
                # r2.backward()
                mse = mean_squared_error(np.array(y), np.array(output1)[:, 0])
                rmse = mean_squared_error(np.array(y), np.array(output1)[:, 0], squared=False)
                r_square = r2_score(np.array(y), np.array(output1)[:, 0])
                mae=mean_absolute_error(np.array(y), np.array(output1)[:, 0])

            train_loss /= batch_size
            train_loss /= len(dataloader.dataset)
            # train_acc = train_correct / len(dataloader.dataset)

        return float(train_loss), float(mse), float(rmse),float(r_square), float(mae)

    # # Load train data
    # try:
    #     with open('/tmp/local_dataset/trainset.pyb', 'rb') as fh:
    #         trainset = pickle.loads(fh.read())
    # except:
    #     # trainset = read_data(trainset=True, nr_examples=settings['training_samples'], data_path='../data/nmrshift.npz')
    #     testset = read_data(data)
    #     try:
    #         if not os.path.isdir('/tmp/local_dataset'):
    #             os.mkdir('/tmp/local_dataset')
    #         with open('/tmp/local_dataset/trainset.pyb', 'wb') as fh:
    #             fh.write(pickle.dumps(trainset))
    #     except:
    #         pass

    # Load test data
    try:
        with open('/tmp/local_dataset/testset.pyb', 'rb') as fh:
            testset = pickle.loads(fh.read())
    except:
        testset = read_data(data)
        # testset = read_data(trainset=False, nr_examples=settings['test_samples'],  data_path='../data/nmrshift.npz')
        try:
            if not os.path.isdir('/tmp/local_dataset'):
                os.mkdir('/tmp/local_dataset')
            with open('/tmp/local_dataset/trainset.pyb', 'wb') as fh:
                fh.write(pickle.dumps(testset))
        except:
            pass


    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=settings['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=settings['batch_size'], shuffle=True)

    try:
        # training_loss, training_acc = evaluate(model, loss, train_loader)
        test_loss, mse1,rmse,r2_s, mae1 = evaluate(model, loss, test_loader)

    except Exception as e:
        print("failed to validate the model {}".format(e), flush=True)
        raise
    
    report = { 
                "classification_report": 'unevaluated',
                # "training_loss": training_loss,
                # "training_accuracy": training_acc,
                "test_loss": test_loss,
                "MSE": mse1,
                "RMSE": rmse,
                "R2_score": r2_s,
                "MAE": mae1,
            }

    print("-- VALIDATION COMPLETE! --", flush=True)
    return report

if __name__ == '__main__':

    with open('settings.yaml', 'r') as fh:
        try:
            settings = dict(yaml.safe_load(fh))
        except yaml.YAMLError as e:
            raise(e)

    from fedn.utils.pytorchhelper import PytorchHelper
    from models.pytorch_model import create_seed_model
    helper = PytorchHelper()
    model, loss, optimizer = create_seed_model(settings)
    print('=========================SADI ======================================= Model Validation')
    model.load_state_dict(np_to_weights(helper.load_model(sys.argv[1])))

    report = validate(model,'../data/test.csv', settings)
    print('=========================Report ======================================= ')
    print(report)

    print('================================================================ ')

    with open(sys.argv[2], "w") as fh:
        fh.write(json.dumps(report))


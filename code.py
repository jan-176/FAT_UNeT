import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil
import click
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
import mlflow.pytorch
from src.models import *
from src.utils import *
from src.pot import *

def backprop(epoch, model, data, feats, optimizer, criterion, scheduler, training=True, _shuffle=False):
    if 'TranAD' in model.name:
        dataset = CustomDataset(data)
        shuffle = True if training and _shuffle else False
        dataloader = DataLoader(dataset, batch_size=model.batch, shuffle=shuffle)
        n = epoch + 1
        l1s = []
        if training:
            model.train()
            for d in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = criterion(z, elem) if not isinstance(z, tuple) else (1 / n) * criterion(z[0], elem) + (1 - 1 / n) * criterion(z[1], elem)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Training at epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.eval()
            test_losses = []
            test_preds = []
            for i, d in enumerate(dataloader):
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                if i == 0:
                    mlflow.log_text(str(summary(model, input_data=(window, elem), verbose=0)), "model_summary.txt")
                z = model(window, elem)
                if isinstance(z, tuple):
                    z = z[1]

                test_loss = criterion(z, elem).detach().cpu().numpy()
                test_pred = z.detach().cpu().numpy()

                test_losses.append(test_loss.squeeze(0))
                test_preds.append(test_pred.squeeze(0))
            return np.concatenate(test_losses, axis=0), np.concatenate(test_preds, axis=0)
    elif 'DTAAD' in model.name:
        _lambda = 0.8
        dataset = CustomDataset(data)
        shuffle = True if training and _shuffle else False
        dataloader = DataLoader(dataset, batch_size=model.batch, shuffle=shuffle)
        l1s = []
        if training:
            for d in dataloader:
                local_bs = d.shape[0]
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, local_bs, feats)
                z = model(window)
                l1 = _lambda * criterion(z[0].permute(1, 0, 2), elem) + (1 - _lambda) * criterion(z[1].permute(1, 0, 2), elem)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            model.eval()
            test_losses = []
            test_preds = []
            for i, d in enumerate(dataloader):
                local_bs = d.shape[0]
                window = d.permute(0, 2, 1)
                elem = window[:, :, -1].view(1, local_bs, feats)

                if i == 0:
                    mlflow.log_text(str(summary(model, input_data=window, verbose=0)), "model_summary.txt")
                z = model(window)
                z = z[1].permute(1, 0, 2)

                test_loss = criterion(z, elem).detach().cpu().numpy()
                test_pred = z.detach().cpu().numpy()

                test_losses.append(test_loss.squeeze(0))
                test_preds.append(test_pred.squeeze(0))
            return np.concatenate(test_losses, axis=0), np.concatenate(test_preds, axis=0)
    elif 'FAT' in model.name:
        shuffle = True if training and _shuffle else False

        # data
        data_enc = data[:, :model.n_enc, :]  # data from 0 to 39 (40)
        data_dec = data[:, model.n_enc - 1:model.n_window - 1, :]  # data from 39 to 48 (10)
        data_labels = data[:, model.n_enc:, :]  # data from 40 to 49 (10)

        dataset = torch.utils.data.TensorDataset(data_enc, data_dec, data_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=model.batch, shuffle=shuffle)

        if training:
            model.train()
            train_losses = []
            for i, (enc, dec, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                outputs = model(enc, dec)
                loss = torch.mean(criterion(outputs, labels))
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tloss= {np.mean(train_losses)}')
            return np.mean(train_losses), optimizer.param_groups[0]['lr']
        else:
            with torch.no_grad():
                model.eval()
                test_losses = []
                test_preds = []
                for i, (enc, dec, labels) in enumerate(dataloader):
                    if i == 0:
                        mlflow.log_text(str(summary(model, input_data=(enc, dec), verbose=0)), "model_summary.txt")
                    outputs = model(enc, dec)
                    test_loss = criterion(outputs, labels).detach().cpu().numpy()[:, -1, :]
                    #test_loss = criterion(outputs, labels).detach().cpu().numpy().mean(axis=1) # new
                    test_pred = outputs.detach().cpu().numpy()[:, -1, :]

                    test_losses.append(test_loss)
                    test_preds.append(test_pred)

                return np.concatenate(test_losses, axis=0), np.concatenate(test_preds, axis=0)
    elif 'UNeT' in model.name:
        shuffle = True if training and _shuffle else False

        # data
        data_enc = data[:, :model.n_enc, :]  # data from 0 to 39 (40)
        data_dec = data[:, model.n_enc - 1:model.n_window - 1, :]  # data from 39 to 48 (10)
        data_labels = data[:, model.n_enc:, :]  # data from 40 to 49 (10)

        dataset = torch.utils.data.TensorDataset(data_enc, data_dec, data_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=model.batch, shuffle=shuffle)

        if training:
            model.train()
            train_losses = []
            for i, (enc, dec, labels) in enumerate(dataloader):
                optimizer.zero_grad()
                output = model(enc, dec).squeeze(1)
                loss = torch.mean(criterion(output, labels[:, -1, :]))
                train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tloss= {np.mean(train_losses)}')
            return np.mean(train_losses), optimizer.param_groups[0]['lr']
        else:
            with torch.no_grad():
                model.eval()
                test_losses = []
                test_preds = []
                for i, (enc, dec, labels) in enumerate(dataloader):
                    if i == 0:
                        mlflow.log_text(str(summary(model, input_data=(enc, dec), verbose=0)),
                                        "model_summary.txt")
                    output = model(enc, dec).squeeze(1)
                    test_loss = criterion(output, labels[:, -1, :]).detach().cpu().numpy()
                    test_pred = output.detach().cpu().numpy()

                    test_losses.append(test_loss)
                    test_preds.append(test_pred)

                return np.concatenate(test_losses, axis=0), np.concatenate(test_preds, axis=0)


def experiment_common(
        model_name: str,
        dataset: str,
        entity: str,
        retrain: bool,
        shuffle: bool,
        val: int,
        mlflow_experiment: str,
        n_epochs: int,
        hyp_lr: float,
        hyp_criterion: str,
        hyp_percentage: float,
        eval_fn: callable,
        params: dict
):
    """Common function for running an experiment.

    Args:
        model_name (str): Model name.
        dataset (str): Dataset name.
        entity (str): Entity to train on.
        retrain (bool): Whether to retrain the model.
        shuffle (bool): Whether to shuffle the data.
        val (int): Frequency of validation.
        mlflow_experiment (str): MLflow experiment name.
        n_epochs (int): Number of epochs.
        hyp_lr (float): Learning rate.
        hyp_criterion (str): Loss function to use.
        hyp_percentage (callable): Evaluation function.
        eval_fn (callable): Evaluation function.
        params (dict): Additional experiment-specific parameters.
    """

    # Early Stopping
    COUNTER = 0
    PATIENCE = 3

    # For TIKI dataset, Validation score is set to infinity as val_score is the loss
    # For other datasets, Validation score is set to 0 as val_score is the F1 score
    BEST_VAL_SCORE = float('inf') if dataset == "TIKI" else 0

    # Path to save the model
    os.makedirs("models", exist_ok=True)
    SAVE_PATH = os.path.join("models", f"{model_name}_{dataset}_{entity}.pt")
    

    # Criterion selection
    if hyp_criterion == "LogCosh":
        criterion = LogCoshLoss()
    elif hyp_criterion == "MSE":
        criterion = nn.MSELoss(reduction='none')
    elif hyp_criterion == "Huber":
        criterion = nn.HuberLoss(reduction='none', delta=params.get('delta', 1.0))
    else:
        raise ValueError("Criterion not found")

    mlflow.set_experiment(mlflow_experiment)
    mlflow.pytorch.autolog()

    with mlflow.start_run(run_name=f"{dataset}_{entity}_{model_name}"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dict_trainD, dict_testD, dict_test_labels, dict_test_timestamps, test_columns = load_dataset(device, dataset=dataset, id=entity, scaler=params["scaler"])

        dims = len(test_columns)
        model, optimizer, scheduler, epoch = load_model(model_name, dims, device=device, lr_d=hyp_lr)
        testO = dict_testD.copy()

        window_size = model.n_window
        trainD = torch.cat([convert_to_windows(d, window_size) for d in dict_trainD.values()], dim=0)

        
        # Experiment 2
        if dataset == "TIKI" and mlflow_experiment == "Experiment_2":
            trainD, valD = train_val_split(data=trainD, train_size=0.8, shuffle=shuffle, percentage=hyp_percentage)
            dict_windowed_valD = {"val": valD} if val > 0 else {}
            dict_windowed_testD = {k: convert_to_windows(v, window_size) for k, v in dict_testD.items()}

        # Experiment 1 and 3
        elif dataset in ["SMD", "SMAP", "MSL"]:
            trainD, _ = train_val_split(data=trainD, train_size=1.0, shuffle=shuffle, percentage=hyp_percentage)

            dict_windowed_testD = {}
            dict_windowed_valD = {}
            dict_val_labels = {}

            #Experiment 3
            if mlflow_experiment == "Experiment_3":
                val_entities = ["machine-1-1", "machine-1-2", "machine-1-3"]
                for k, v in dict_testD.items():
                    windowed_data = convert_to_windows(v, window_size)
                    if k in val_entities:
                        dict_windowed_valD[k] = windowed_data
                        dict_val_labels[k] = dict_test_labels[k]
                        dict_test_labels.pop(k)
                    else:
                        dict_windowed_testD[k] = windowed_data
            # Experiment 1 
            else:
                dict_windowed_testD = {k: convert_to_windows(v, window_size) for k, v in dict_testD.items()}

        # save parameters to mlflow
        mlflow.log_param("model", model_name)
        mlflow.log_param("window_size", window_size)
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("entity", entity)
        mlflow.log_param("device", device)
        mlflow.log_param("epochs", n_epochs)
        mlflow.log_param("lr", hyp_lr)

        # Additional parameters passed in the dictionary
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Training phase
        if retrain:
            for e in tqdm(range(n_epochs)):
                lossT, lr = backprop(e, model, trainD, dims, optimizer, criterion, scheduler, training=True,
                                     _shuffle=shuffle)
                mlflow.log_metric("train_loss", lossT, step=e)
                mlflow.log_metric("train_lr", lr, step=e)

                # Validation phase
                if e > 0 and val > 0 and e % val == 0:
                    model.eval()
                    val_score = 0

                    # Experiment 3 uses lossT as init_score
                    lossT,_ = backprop(e, model, trainD, dims, optimizer, criterion, scheduler, training=False, _shuffle=shuffle)
                    params["init_score"] = np.mean(lossT, axis=1)

                    for i, (val_file, valD) in enumerate(dict_windowed_valD.items()):
                        lossV,_ = backprop(e, model, valD, dims, optimizer, criterion, scheduler, training=False, _shuffle=shuffle)
                        if dataset != "TIKI":
                            mean_lossV = np.mean(lossV, axis=1)
                            labelsFinal = (np.sum(dict_val_labels[val_file], axis=1) >= 1) + 0

                            result = eval_fn(mean_lossV, labelsFinal, params)
                            result_f1 = result['final_result']['f1']
                            mlflow.log_metric(f"val_f1_{val_file}", result_f1, step=e)
                            val_score += result_f1
                        else:
                            val_score += np.mean(lossV)
                            # log to mlflow the loss for each validation file
                            mlflow.log_metric("val_loss", val_score, step=e)


                    val_score /= len(dict_windowed_valD)

                    # Determine the condition for early stopping
                    keep_going = val_score < BEST_VAL_SCORE if dataset == "TIKI" else val_score > BEST_VAL_SCORE

                    # EARLY STOPPING
                    if keep_going:
                        BEST_VAL_SCORE = val_score
                        torch.save(model.state_dict(), SAVE_PATH)
                        COUNTER = 0
                    else:
                        COUNTER += 1
                        print(f"Early stopping: {COUNTER}/{PATIENCE}")
                        if COUNTER == PATIENCE:
                            print(f'Early stopping at epoch {e + 1}')
                            break
                    model.train()

        # Load the best model from the training phase for testing
        if n_epochs > 1 and val > 0:
            model.load_state_dict(torch.load(SAVE_PATH))
            model.to(device)
        model.eval()

        lossT,_ = backprop(0, model, trainD, dims, optimizer, criterion, scheduler, training=False, _shuffle=shuffle)

        print("Testing phase")
        for i, (test_file, testD) in tqdm(enumerate(dict_windowed_testD.items())):
            test_labels = dict_test_labels[test_file]
            loss, y_pred = backprop(0, model, testD, dims, optimizer, criterion, scheduler, training=False, _shuffle=shuffle)

            ### Scores
            df = pd.DataFrame()
            test_pot_predictions = []
            test_pot_thresholds = []

            for j in range(dims):
                test_col_name = test_columns[j]
                params["init_score"] = lossT[:, j]
                result = eval_fn(loss[:, j], test_labels[:, j], params)

                # save the pot predictions and thresholds
                test_pot_predictions.append(result.get("final_pot_predictions", None))
                test_pot_thresholds.append(result.get("final_pot_thresholds", None))

                result_df = pd.DataFrame([result.get("final_result", {})])
                result_df["column"] = test_col_name
                df = pd.concat([df, result_df], ignore_index=True)

            mlflow.log_metric(f"test_loss_{test_file}", np.mean(loss))

            # Final results
            params["init_score"] = np.mean(lossT, axis=1)
            mean_loss = np.mean(loss, axis=1)
            labelsFinal = (np.sum(test_labels, axis=1) >= 1) + 0

            test_result = eval_fn(mean_loss, labelsFinal, params)

            final_result = test_result["final_result"]
            final_result.update(hit_att(loss, test_labels))
            final_result.update(ndcg(loss, test_labels))

            final_pot_predictions = test_result.get("final_pot_predictions", None)
            final_pot_thresholds = test_result.get("final_pot_thresholds", None)

            # make second result dataframe for Final results
            final_result_df = pd.DataFrame([final_result])

            np_pot_predictions = np.array(test_pot_predictions).T
            np_pot_thresholds = np.array(test_pot_thresholds).T
            np_test_data = testO[test_file].cpu().numpy()

            with tempfile.TemporaryDirectory() as tmpdirname:

                # make folder for test file and model name
                save_folder = f"{tmpdirname}/{model_name}_{test_file}"
                os.makedirs(save_folder, exist_ok=True)

                # save df to csv
                df.to_csv(f'{save_folder}/test_results.csv', index=False)

                # save model
                torch.save(model.state_dict(), f"{save_folder}/model.pt")

                # save final result df to csv
                final_result_df.to_csv(f'{save_folder}/final_results.csv', index=False)

                # save pot_predictions to npy
                np.save(f"{save_folder}/pot_prediction.npy", np_pot_predictions)

                # save pot_thresholds to npy
                np.save(f"{save_folder}/pot_threshold.npy", np_pot_thresholds)

                # save final_pot_predictions to npy
                np.save(f"{save_folder}/final_pot_predictions.npy", final_pot_predictions)

                # save final_pot_thresholds to npy
                np.save(f"{save_folder}/final_pot_thresholds.npy", final_pot_thresholds)

                # save y_pred to npy
                np.save(f"{save_folder}/y_pred.npy", y_pred)

                # save loss to npy
                np.save(f"{save_folder}/loss.npy", loss)

                # save timestamps to npy
                np.save(f"{save_folder}/timestamps.npy", dict_test_timestamps[test_file])

                # save test data to npy
                np.save(f"{save_folder}/test_data.npy", np_test_data)

                # save the labels to npy
                np.save(f"{save_folder}/labels.npy", dict_test_labels[test_file])

                # make zip folder of "test_file" folder
                shutil.make_archive(save_folder, 'zip', save_folder)

                # log zip folder to mlflow
                mlflow.log_artifact(f"{save_folder}.zip")

            pre_metric_name = f"test_{test_file}" if len(dict_testD) > 1 else "test_"
            for key, value in final_result.items():
                metric_name = f"{pre_metric_name}_{key}".replace("@", "_at_").replace("%", "pct")
                mlflow.log_metric(metric_name, value)

            mlflow.log_metric(f"{pre_metric_name}_f1_new", calculate_f1_score(df))
        mlflow.end_run()


def eval_fn_exp_1_2(loss, labels, params):
    """Evaluates the model using the POT-based evaluation and additional params .

    Args:
        loss (np.ndarray): The loss values from the model.
        labels (np.ndarray): The true labels.
        params (dict): Additional parameters for the evaluation.

    Returns:
        dict: The evaluation results including metrics and predictions.
    """
    # Perform POT-based evaluation
    final_result, final_pot_predictions = pot_eval(params.get('init_score', np.array([])), loss, labels, lm=(params.get('lm_d0', None), params.get('lm_d1', None)))

    return {
        "final_result": final_result,
        "final_pot_predictions": final_pot_predictions
    }
def eval_fn_exp_3(loss, labels, params):
    """Evaluates the model using the POT-based evaluation and additional metrics.

    Args:
        loss (np.ndarray): The loss values from the model.
        labels (np.ndarray): The true labels.
        params (dict): Additional parameters for the evaluation.

    Returns:
        dict: The evaluation results including metrics and predictions.
    """
    # Perform POT-based evaluation
    final_result, final_pot_predictions, final_pot_thresholds = pot_eval_dynamic(loss[:1000], loss[1000:], labels[1000:], q=params.get('q', None), level=params.get('level', None))

    return {
        "final_result": final_result,
        "final_pot_predictions": final_pot_predictions,
        "final_pot_thresholds": final_pot_thresholds
    }

# Main CLI
@click.group()
def cli():
    """Main CLI for running experiments."""
    pass

@cli.command('experiment1_2')
@click.option('--config', type=click.Path(exists=True), is_eager=True, help="Path to a JSON config file with parameters.")
def experiment1_2(config):
    """
    A function to run the 3_1 experiment with the given model and dataset configuration.
    Args are dynamically passed to the `experiment_common` function.
    """

    # Load the parameters from the config file
    params = json.load(open(config))

    # Set up additional parameters as a dictionary
    additional_params = {
        'lm_d0': params.get("hyp_lm_d0"),
        'lm_d1': params.get("hyp_lm_d1"),
        'delta': params.get("hyp_delta"),
        'scaler': params.get("scaler", "min_max")
    }
    print(params)

    # Call the common experiment function with evaluation and saving functions
    experiment_common(params.get("model_name"), params.get("dataset"), params.get("entity"), params.get("retrain"), params.get("shuffle"), 
                      params.get("val"), params.get("mlflow_experiment"), params.get("n_epochs"), params.get("hyp_lr"), params.get("hyp_criterion"), 
                      params.get("hyp_percentage"), eval_fn_exp_1_2, additional_params)

@cli.command('experiment3')
@click.option('--config', type=click.Path(exists=True), is_eager=True, help="Path to a JSON config file with parameters.")
def experiment3_1(config):
    """
    A function to run the experiment 3 with the given model and dataset configuration.
    Args are dynamically passed to the `experiment_common` function.
    """

    # Load the parameters from the config file
    params = json.load(open(config))

    # Set up additional parameters as a dictionary
    additional_params = {
        'q': params.get('hyp_q'),
        'level': params.get('hyp_level'),
        'delta': params.get('hyp_delta'),
        'scaler': params.get("scaler", "min_max")
    }

    # Call the common experiment function with evaluation and saving functions
    experiment_common(params.get("model_name"), params.get("dataset"), params.get("entity"), params.get("retrain"), params.get("shuffle"), 
                      params.get("val"), params.get("mlflow_experiment"), params.get("n_epochs"), params.get("hyp_lr"), params.get("hyp_criterion"), 
                      params.get("hyp_percentage"), eval_fn_exp_3, additional_params)

if __name__ == '__main__':
    cli()

# Using GaMPEN
On this page, we go over the most important user-facing functions that GaMPEN has and what each of these functions do. Read this page carefully to understand the various arguments/options that can be set while using these functions. 


## Make Splits

```{eval-rst}
:py:mod:`ggt.data.splits`
=========================

.. py:module:: ggt.data.splits


Functions
~~~~~~~~~

.. py:function:: build(z_bin, band, seed=42, ...)

   Create a seeded train/devel/test split and a fitted label scaler, or
   verify and reuse the existing ones.

```

This `GaMPEN/ggt/data/splits.py` script partitions `info.csv` into
`train`/`devel`/`test` and fits the label scaler, writing both into a
`splits/` folder beside it.

**It replaces the earlier `make_splits.py`**, which generated fourteen slug
variants on every invocation with a hardcoded `random_state=0`. The
contract here is different in three ways that matter for reproducibility:

* **A split is created once and then reused.** On every later call the
  split's manifest hash is verified and the existing files are loaded.
  Regenerating one requires `--force-resplit` *and* a new `--seed`, so a
  hyperparameter sweep cannot silently reshuffle the test set underneath a
  series of runs.
* **The split is checked against the current `info.csv`.** Hashes alone
  only prove a CSV has not been hand-edited. Rebuilding `info.csv` -- a
  retuned quality cut, or a move from a debug subset to production data --
  leaves the previous split perfectly self-consistent and completely wrong,
  so the union of its object ids must still equal the catalog's.
* **The scaler is fitted once, on the train split only, and persisted** to
  `splits/<slug>-scaler.joblib`, rather than being refitted invisibly on
  every dataset construction.

Balancing is retained: with `--balance-on` set, rows are partitioned into
four quantile bins of that column and interleaved, so each split spans the
full range of the balancing variable.

Splits are named `<slug>-<split>.csv` where the slug is `euclid-<seed>`, so
different seeds coexist and each stays independently verifiable.

### Parameters:
* **--z-bin** (*int*, required) - Redshift bin index.

* **--band** (*str*, required) - Which band's `info.csv` to split.

* **--seed** (*int*, default=`42`) - Split seed; also names the slug.

* **--train-frac / --devel-frac / --test-frac** (*float*, default=`0.70` /
  `0.15` / `0.15`) - Must sum to 1.

* **--balance-on** (*str*, default=`"bt"`) - Column used for quantile
  balancing.

* **--force-resplit** (*flag*) - Refused on an existing seed; choose a new
  one instead.

* **--out-root** (*str*, default=`None`) - Override the data root.


## Running the trainer

`GaMPEN/ggt/train/train.py` is invoked from the command line and runs one
fine-tune end to end: resolve paths, verify the pixel convention, create or
reuse the split and scaler, build the datasets, load a pretrained
checkpoint, train, and write the diagnostic figures.

```{eval-rst}

:py:mod:`ggt.train.train`
=========================

.. py:module:: ggt.train.train

Functions
~~~~~~~~~

.. py:function:: main(**kwargs)

   Fine-tune a model on one (z_bin, band) dataset.

.. py:function:: run(args)

   The same pipeline, callable from Python with an attribute-style object.

```

Each stage is a separate importable function -- `resolve`,
`check_pixel_zp`, `prepare_split`, `build_loaders`, `build_net`,
`build_optimizer` -- so an analysis notebook can rebuild exactly the
datasets and model a run used without training anything.

`python -m ggt.train.train --help` is authoritative; the options below are
the ones whose *meaning* is not obvious from the name.

### Parameters:

* **--z-bin** (*int*, required) - Redshift bin index. Selects the dataset,
  the crop size, and which published checkpoint initialises the model.

* **--band** (*str*, required) - Which band to train on.

* **--run-name** (*str*, required) - Names the run directory and the log.

* **--cutout-size** (*int*, default=`None`) - What the network sees.
  Defaults to the bin's `target_crop_px`. May be smaller than the cache's
  `crop_px` but never larger.

* **--pixel-zp** (*str*, default=`"none"`) - The photometric convention the
  pixels are on. **This is checked against the cache manifest and the run
  refuses to start on a mismatch.** Training on differently-scaled pixels
  than you believe produces a model that trains perfectly well and means
  nothing.

* **--init-from** (*str*, default=`"real"`) - `real`, `sim` or `scratch`.
  Which published checkpoint family to transfer from.

* **--freeze** (*str*, default=`"vgg_features_early"`) - Which parameter
  group to freeze. Note the parameter budget is lopsided: the classifier is
  ~74% of the model and the convolutional features only ~9%, so freezing
  the features frees far less than it appears to.

* **--allow-broad-reinit** (*flag*) - By default only `fc_loc.0.weight` may
  be re-initialised when loading a checkpoint, because it is the sole
  input-size-dependent tensor. Anything else being re-initialised means the
  transfer is not doing what you think, so it raises. This flag disables
  that guard.

* **--head-lr-mult** (*float*, default=`10.0`) - The re-initialised STN head
  and the classifier train at this multiple of `--lr`; the backbone is
  already close to right.

* **--lr** (*float*, default=`5e-7`) - `aleatoric_cov` exponentiates its
  variance terms and diverges easily. Anything above ~1e-6 should be
  treated as suspect.

* **--patience** (*int*, default=`25`) - Early-stopping patience on devel
  loss. `ReduceLROnPlateau` uses a third of it.

* **--expand-data** (*int*, default=`4`) - Augmentation factor, applied to
  the train split only. The per-epoch train *evaluation* uses an
  un-augmented view, because a loss measured through random rotations is
  not comparable with the devel loss.

* **--force-resplit** (*flag*) - Refused on an existing seed; pick a new
  `--seed` instead.

* **--limit** (*int*, default=`None`) - Truncate every split. For overfit
  and smoke tests.

* **--figures-dir** (*str*, default=`None`) - Where figures go. Defaults to
  `training_eval_figs/` inside the run directory.

* **--mlflow** (*flag*) - Also log to MLflow. Off by default: on MLflow 3.x
  the `file://` store silently records nothing at all, so the flat
  `metrics.csv` and `train.log` are the source of truth. Point it at a
  sqlite tracking URI if you turn this on.

### What a run writes

Everything a run produces sits in one directory on the data volume, so a
run can be copied or archived whole. A `best.pt`/`last.pt` pair is over a
gigabyte, which is why this is never inside the package.

```
<runs_root>/<bin>/<band>/<run_name>/
    config.json     every resolved argument, both git SHAs, the split
                    hashes and the scaler statistics -- enough to
                    reconstruct the run from this file alone
    metrics.csv     one row per epoch, flushed as it goes
    train.log       one human-readable line per epoch, for tail -f
    best.pt         lowest devel loss so far
    last.pt         rewritten every epoch, so a killed run is resumable
    training_eval_figs/    ten diagnostic figures
```

## Inference

```{eval-rst}
:py:mod:`ggt.modules.inference`
===============================

.. py:module:: ggt.modules.inference

Functions
~~~~~~~~~~

.. py:function:: main(model_path, output_path, data_dir, cutout_size, channels, parallel, slug, split, normalize, batch_size, n_workers, label_cols, model_type, repeat_dims, label_scaling, mc_dropout, dropout_rate, transform, errors, cov_errors, n_runs, ini_run_num)

```
The `GaMPEN/ggt/modules/inference.py` script provides users the functionality to perform predictions on images using trained GaMPEN models.

### Parameters

* **model_type** (*str*; default=`"vgg16_w_stn_oc_drp"`) - Same as the model types mentioned in [Running the Trainer](#running-the-trainer) section previously. If using our pre-trained models, this should be set to `vgg16_w_stn_oc_drp`.

* **model_path** (*str*; required variable)- The full path to the trained `.pt` model file which you want to use for performing prediction.

    ```{attention}
    The model path should be enclosed in single quotes `'/path/to/model/xxxxx.pt'` and NOT within double quotes `"/path/to/model/xxxxx.pt"`. If you encluse it within double quotes, then the script will throw up an error.
    ```

* **output_path** (*str*; required variable) - The full path to the output directory where the predictions of the model will be stored.

* **data_dir** (*str*; required variable) - The full path to the data directory that should contain a `cutouts` folder with all the images that you want to perform predictions on as well as an `info.csv` file that contains the filenames for all the images. For more information on how to create this directory structure during performing inference, plese refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions)

* **cutout_size** (*int*; default=`167`) - Size of the input image that the model takes as input. For our pre-trained models, this should be set to `239`, `143`, `96` for the low, mid, and high redshift models respectively.

* **channels** (*int*; default=`3`) - Number of channels in the input image. For our pre-trained models, this should be set to `3`.

* **slug** (*str*; required variable) - This specifies which slug (balanced/unbalanced xs, sm, lg, dev, dev2) is used to perform predictions on. Each slug refers to a different way to split the data into train, devel, and test sets. For consistent results, you should set this to the same slug that was used to train the model. For more information on the fraction of data assigned to the train/deve/test sets for each slug, please refer to the [`make_splits`](#make-splits) function.

    If you are performing predictions on a dataset for which you don't have access to the ground truth labels (and thus you haven't run `make_splits`), this should be set to `None` as shown in the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions).

* **split** (*str*; default=`"test"`) - The split of the data that you want to perform predictions on. This should be set to `test` if you are performing predictions on the test set. If you are performing predictions on the train or devel set, this should be set to `train` or `devel` respectively.

    If you are performing predictions on a dataset for which you don't have access to the ground truth labels (and thus you haven't run `make_splits`), this should be set to `None` as shown in the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions).

* **normalize/no-normalize** (*bool*; default=`True`) - The normalize argument controls whether or not, the loaded images will be normalized using the `arsinh` function. This should be set to the same value as what was used during training the model.

* **label_scaling** (*str*; default=`"std"`) - The label scaling option controls whether to perform an inverse-transformation on the predicted values. Set this to `std` for [sklearn's `StandardScaling()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) and `minmax` for [sklearn's `MinMaxScaler()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html). This should usually always be set to `std` especially when using multiple target variables.
    
    Note that you should pass the same argument for `label_scaling` as was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `std`."

* **batch_size** (*int*; default=`256`) - The batch size to be used during inference. This specfies how many images will be processed in a single batch. During inference, the only consideration is to keep the batch size small enough so that the batch can be fit within the memory of the GPU.

* **n_workers** (*int*; default=`4`) - The number of workers to be used during the
data loading process. You should set this to the number of threads you have access to. 

* **parallel/ no-parallel** (*bool*; default=`True`) - The parallel argument controls whether or not to use multiple GPUs when they are available. 

    Note that this variable needs to be set to whatever value was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `parallel`

* **label_cols** (*str*; default=`bt_g`) - Enter the label column(s) separated by commas. Note that you should pass the exactly same argument for label_cols as was used during the training phase (of the model being used for inference)

* **repeat_dims/no-repeat_dims** (*bool*; default=`True`) - In case of multi-channel data, whether to repeat a two dimensional image as many times as the number of channels. Note that you should pass the exactly same argument for `repeat_dims` as was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `repeat_dims`

* **mc_dropout/no-mc_dropout** (*bool*; default=`True`) - Turns on Monte Carlo dropout during inference. For most cases, this should be set to `mc_dropout`.

* **n_runs** (*int*; default=`1`) - The number of different models that will be generated using Monte Carlo dropout and used for infererence. 

* **ini_run_num** (*int*; default=`1`) - Specifies the starting run-number for `n_runs`. For example, if `n_runs` is set to 5 and `ini_run_num` is set to 10, then the output csv files will be named as `inf_10.csv`, `inf_11.csv`, `inf_12.csv`, `inf_13.csv`, `inf_14.csv`.

* **dropout_rate** (*float*; default=`None`) - This should be set to the dropout rate that was used while training the model.

* **transform/no-transform** (*bool*; default=`False`) - If `True`, the images are passed through a cropping transformation to ensure proper cutout size. This should be left on for most cases.

   ```{attention}
   Note that if you set this to True and then use cutouts which have a smaller size than the `cutout_size`, this will lead to unpredictable behaviour.
   ```

* **errors/no-errors** (*bool*, default=`False`) - If True and if the model allows for it, aleatoric uncertainties are written to the output file. Only set this to True if you trained the model with `aleatoric` loss. 

* **cov_errors/no-cov_errors** (*bool*, default=`False`) - If True and if the model allows for it, aleatoric uncertainties with full covariance conisdered are written to the output file. Only set this to True if you trained the model with `aleatoric_cov` loss. For our pre-trained models, this should be set to `cov_errors`.

* **labels/no-labels** (*bool*, default=`True`)- If True, this means you have labels available for the dataset. If False, this means that you have no labels available and want to perform predictions on a dataset for which you don't know the ground truth labels.

    This primarily used to control which files are used to perform scaling the prediction variables. If `--no-labels`, then you need to specify the data directory and slug that should be used to perform the scaling. If `--labels`, then the  `scaling_data_dir` and `scaling_slug` are automatically set to values for `data_dir` and `slug` provided before. 

* **scaling_data_dir** (*str*; default=`None`) - The data directory that should be used to perform unscaling of the prediction variables. You should only set this if using `--no-labels`.

    This scaling refers to the `label_scaling` variable that you passed before. Essentially to inverse transform the predictions, we need access to the original scaling parameters that were used to scale the data during trianing. In case you are using a pre-trained model directly on some data for which you have no labels, you need to point this to the `/splits` folder of the data-directory that was used to train the model. For all our pre-trained models, we make the relevant scaling files available. Refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions) for a demonstration. 

* **scaling_slug** (*str*; default=`None`) - This needs to be set only if you are using `--no-labels`. This specifies which slug (`balanced/unbalanced`, `xs`, `sm`, `lg`, `dev`,`dev2`) corresponding to the scaling_data_dir that should be used to perform the data scaling on. 

    For example, if you want a `balanced-dev2-train.csv` file in the `scaling_data_dir` , then you should set this to `balanced-dev2`. Refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions) for a demonstration.



## Result Aggregator

The `GaMPEN/ggt/modules/result_aggregator.py` module is used to aggregate the prediction `.csv` files generated by the  [inference module](#inference). 

```{attention}
The unsccaling properties of the `result_aggregator` module is mostly useful when you are predicting variables similar to the ones used in [Ghosh et. al. 2022](https://doi.org/10.3847/1538-4357/ac7f9e).

If you are using your own custom scaling of variables (or predicting other variables), then you will need to run the `result_aggregator` module with `--no-unscale` and perform the unscaling of variables yourself. Alternatively, you can also choose to alter the `unscale_preds` function in the `result_aggregator.py` module to suit your needs.
```


```{attention}
The result aggregator module also converts flux to magnitudes. However, this conversion is only valid for HSC. If you are using the module for some other survey, please alter the magnitude conversion line in the `unscale_preds` function of `result_aggregator.py` or ignore the mangitudes produced by the `result_aggregator` module.
```


```{eval-rst}
:py:mod:`ggt.modules.result_aggregator`
===============================

.. py:module:: ggt.modules.result_aggregator

Functions
~~~~~~~~~~

.. py:function:: main(data_dir,num,out_summary_df_path,out_pdfs_path, unscale,scaling_df_path, drop_old)

```

### Parameters

* **data_dir** (*str*; required variable)  - Full path to the directory that has the prediction csv files that need to be aggregated.

* **num** (*int*; default=`500`) - The number of prediction csv files that need to be aggregated.

* **out_summary_df_path** (*str*; required variable) - Full path to the output csv file that will contain the summary statistics. 

* **out_pdfs_path** (*str*; required variable) - Full path to the output directory that will contain the posterior distribution functions (PDFs) of the predicted output variables for each galaxy.

* **unscale/no-unscale** (*bool*, default=`False`) - If `True`, the predictions are unscaled using the information `scaling_df_path`. This unscaling is for the inverse logit and logarithmic tansformations (e.g., converting $\log R_e$ to $R_e$)

    This is only useful if you are using our pre-trained models/you are predicted the same variables as in [Ghosh et. al. 2022](https://doi.org/10.3847/1538-4357/ac7f9e). For all other cases, if you want to set this to `True`, you will need to modify the `unscale_preds` function in `result_aggregator.py` according to the variables you are predicting and the transformations you made to them during training.  

    ```{attention}
    In order to make sure that you are not making a mistake, the module will throw an error if you are using the `--unscale` option and the inference `.csvs` do not have the column names exactly as is expected for our trained models (i.e., `custom_logit_bt`,`ln_R_e_asec`,`ln_total_flux_adus`).

    When you are using using some different scaling, you need to run this script with `--no-unscale` and transform the predictions yourself. Or you can also alter the `unscale_preds` function in `result_aggregator.py` according to your needs.
    ``` 

* **scaling_df_path** (*str*; default=`None`) - Full path to the `info.csv` file that contains the scaling information. This is only used if `unscale` is set to True. 

    This is needed to perform the inverse logit transforamtion. As the logit transformation goes to infinty at the edges of the variable space and we need to perform an approximation. To perform this approximation, we need access to the `info.csv` file that was used during training. We make the `info.csv` files for all our pre-trained models available. Refer to the [Predictions Tutorial](https://gampen.readthedocs.io/en/latest/Tutorials.html#making-predictions) for a demonstration.

* **drop_old/no-drop_old** (*bool*; default=`True`)- If `True`, the unscaled prediction columns will be dropped.



## AutoCrop

```{eval-rst}
:py:mod:`ggt.modules.autocrop`
===============================

.. py:module:: ggt.modules.autocrop

Functions
~~~~~~~~~~

.. py:function:: main(model_type, model_path, cutout_size, channels, n_pred,image_dir, out_dir, normalize, transform, repeat_dims, parallel, cov_errors,errors,)

```
The `GaMPEN/ggt/modules/autocrop.py` script provides users the functionality to perform cropping using a trained GaMPEN model and then save these cropped images as fits files for further analysis.

### Parameters

* **model_type** (*str*; default=`"vgg16_w_stn_oc_drp"`) - Same as the model types mentioned in [Running the Trainer](#running-the-trainer) section previously. If using our pre-trained models, this should be set to `vgg16_w_stn_oc_drp`.

* **model_path** (*str*; required variable)- The full path to the trained `.pt` model file which you want to use for performing prediction.

    ```{attention}
    The model path should be enclosed in single quotes `'/path/to/model/xxxxx.pt'` and NOT within double quotes `"/path/to/model/xxxxx.pt"`. If you encluse it within double quotes, then the script will throw up an error.
    ```
* **cutout_size** (*int*; default=`167`) - Size of the input image that the model takes as input. For our pre-trained models, this should be set to `239`, `143`, `96` for the low, mid, and high redshift models respectively.

* **channels** (*int*; default=`3`) - Number of channels in the input image. For our pre-trained models, this should be set to `3`.

* **n_pred** (*int*; default=`1`) - Number of output variables that were used while training the model.

* **image_dir** (*str*; required variable) - Full path to the directory that contains the images that need to be cropped.

* **out_dir** (*str*; required variable) - Full path to the directory where the cropped images will be saved.

* **normalize/no-normalize** (*bool*; default=`True`) - The normalize argument controls whether or not, the loaded images will be normalized using the `arsinh` function. This should be set to the same value as what was used during training the model.

* **transform/no-transform** (*bool*; default=`True`) - The transform argument controls whether or not, the loaded images will be cropped to the mentioned `cutout_size` while being loaded. This should be set to `True` for most cases.

* **repeat_dims/no-repeat_dims** (*bool*; default=`True`) - In case of multi-channel data, whether to repeat a two dimensional image as many times as the number of channels. Note that you should pass the exactly same argument for `repeat_dims` as was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `repeat_dims`

* **parallel/ no-parallel** (*bool*; default=`True`) - The parallel argument controls whether or not to use multiple GPUs when they are available. 

    Note that this variable needs to be set to whatever value was used during the training phase (of the model being used for inference). For all our pre-trained models, this should be set to `parallel`

* **errors/no-errors** (*bool*, default=`False`) - If True and if the model allows for it, aleatoric uncertainties are written to the output file. Only set this to True if you trained the model with `aleatoric` loss. 

* **cov_errors/no-cov_errors** (*bool*, default=`False`) - If True and if the model allows for it, aleatoric uncertainties with full covariance conisdered are written to the output file. Only set this to True if you trained the model with `aleatoric_cov` loss. For our pre-trained models, this should be set to `cov_errors`.



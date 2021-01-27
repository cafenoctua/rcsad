# kaggle-template

# local env
Use Docker in your local environment to access both Python and Jupyter environments.<br/>
For the Docker image, pull the image provided by kaggle and use it.<br/>
There are two types, CPU and GPU, but we'll start with the CPU version.<br/>
Here is the github repository provided by kaggle.<br/>
https://github.com/Kaggle/docker-python
# About kaggle api
[kaggle-api](https://github.com/Kaggle/kaggle-api)

## Download competitions data　and dataset
```
kaggle competitions download {competitions name}
```


Displayed in the URL of the competition page with the name of the competition specified in kaggle-api.<br/>
Example:`https://www.kaggle.com/c/{competition name}`

The same is true for datasets: `https://www.kaggle.com/{user name}/{dataset name}`, where {user name} and {dataset name} are used to download the dataset.

```
kaggle dataset download {user name}/{dataset name}
```

## Upload file to kaggle
Submit submission.csv to competition.
```
kaggle competitions submissions {competition name} -f submission.csv -m "messasge."
```

## Show same lists
Show lists competitions
```
kaggle competitions list
```

Add -h options after list you can see list help.

Sort example:

`kaggle competitions list --sort-by 'prize'`

Show kernels list.
```
kaggle kernels list -s titanic
```
# import libraries and load data


```python
import cv2 as cv
import os
import numpy as np
```


```python
data = []
label = []
IMG_SIZE = 32
prefix_path = "./Train/"
sub_paths = [str(i) for i in range(43)]
for sub_path in sub_paths:
    path = prefix_path + sub_path +"/"
    number = 0
    for file in os.listdir(path):
        img = cv.imread(path + file)
        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32')
        label.append(int(sub_path))
        data.append(img)
        number = number+1
        if number > 199:
            break
data = np.array(data)
print(data.shape)
```

    (8600, 32, 32, 3)



```python
data_label = np.array([int(i) for i in label])
```


```python
# Data Normalization
# Normalization
from sklearn.preprocessing import StandardScaler
# data = data/255.0
reshaped_data = data.reshape(len(data),-1)
reshaped_data = StandardScaler().fit_transform(reshaped_data)
reshaped_data.shape
```




    (8600, 3072)




```python
print(reshaped_data.shape)
```

    (8600, 3072)


# Each picture is a 3072-length vector


```python
from clustimage import Clustimage
import pandas as pd
Xraw = reshaped_data
filenames = list(map(lambda x: str(x) + '.png', np.arange(0, reshaped_data.shape[0])))
Xraw = pd.DataFrame(Xraw, index=filenames)
print(Xraw)
```

                  0         1         2         3         4         5     \
    0.png     1.020967  1.025078  1.268246  0.274916  0.505294  1.032035   
    1.png    -0.023040  0.790790  0.526703 -0.137457  0.700525  0.394440   
    2.png    -0.242152 -0.172394  0.071369 -0.330757 -0.262612 -0.113033   
    3.png    -0.293708 -0.406682 -0.253869 -0.266324 -0.340704 -0.269179   
    4.png     0.312074  0.296182  0.305541  0.326463  0.297048  0.290343   
    ...            ...       ...       ...       ...       ...       ...   
    8595.png -0.951046 -1.005418 -1.021431 -0.962204 -1.017503 -1.049907   
    8596.png -0.912379 -0.979386 -0.995412 -0.910658 -0.978457 -1.010870   
    8597.png -0.886601 -0.927322 -0.917355 -0.936431 -0.978457 -0.984846   
    8598.png -0.886601 -0.862242 -0.826288 -0.897771 -0.874334 -0.854725   
    8599.png -0.886601 -0.888274 -0.865316 -0.884884 -0.874334 -0.867737   
    
                  6         7         8         9     ...      3062      3063  \
    0.png    -0.017754  0.376545  1.052711  0.030038  ...  0.585032  0.483126   
    1.png    -0.210100  0.609969  0.302718 -0.072416  ... -0.359648 -0.261513   
    2.png    -0.299862 -0.258886 -0.175727 -0.341358  ... -0.405358 -0.370106   
    3.png    -0.274216 -0.349662 -0.279174 -0.302938  ...  0.569795  0.700312   
    4.png     0.302823  0.285769  0.289787  0.247752  ...  1.255450  1.398410   
    ...            ...       ...       ...       ...  ...       ...       ...   
    8595.png -0.953839 -1.011029 -1.042099 -0.956081  ... -0.923408 -0.819992   
    8596.png -0.915370 -0.972125 -1.003306 -0.904854  ... -0.892935 -0.851018   
    8597.png -0.953839 -1.011029 -1.029168 -0.943274  ... -0.908172 -0.757938   
    8598.png -0.889723 -0.881349 -0.861066 -0.892048  ... -0.496779 -0.804478   
    8599.png -0.902547 -0.894317 -0.873997 -0.904854  ... -0.755804 -0.726912   
    
                  3064      3065      3066      3067      3068      3069  \
    0.png     0.501985  0.451423  0.610877  0.643794  0.591087  0.564290   
    1.png    -0.284007 -0.345836 -0.196248 -0.156922 -0.128063 -0.271847   
    2.png    -0.345654 -0.299840 -0.304899 -0.264711 -0.219869 -0.240879   
    3.png     0.579043  0.298104  0.579834  0.505209  0.223861  0.920423   
    4.png     1.287977  1.172022  1.200699  1.074949  0.881807  1.075263   
    ...            ...       ...       ...       ...       ...       ...   
    8595.png -0.885060 -0.913116 -0.848156 -0.896045 -0.908416 -0.860240   
    8596.png -0.900472 -0.913116 -0.848156 -0.911443 -0.923717 -0.844756   
    8597.png -0.823414 -0.851789 -0.770548 -0.834451 -0.847212 -0.813788   
    8598.png -0.730944 -0.621810 -0.770548 -0.726663 -0.663599 -0.767336   
    8599.png -0.730944 -0.744465 -0.739505 -0.726663 -0.740105 -0.736368   
    
                  3070      3071  
    0.png     0.578423  0.524764  
    1.png    -0.187958 -0.114829  
    2.png    -0.203285 -0.145286  
    3.png     0.854320  0.600906  
    4.png     0.930958  0.585678  
    ...            ...       ...  
    8595.png -0.908355 -0.921936  
    8596.png -0.908355 -0.921936  
    8597.png -0.862373 -0.876250  
    8598.png -0.663114 -0.647824  
    8599.png -0.709096 -0.723966  
    
    [8600 rows x 3072 columns]


# Parameters for clustering


```python
cl = Clustimage(method='pca',
                embedding='umap',
                grayscale=True,
                dim=(32,32),
                params_pca={'n_components':0.95},
                verbose=50)
results = cl.fit_transform(Xraw,
                           cluster='kmeans',
                           evaluate='silhouette',
                           metric='cosine',
                           linkage='ward',
                           min_clust=40,
                           max_clust=45,
                           cluster_space='high')
```

    [pca] >Column labels are auto-completed.
    [pca] >The PCA reduction is performed to capture [95.0%] explained variance using the [3072] columns of the input data.
    [pca] >Fit using PCA.
    [pca] >Compute loadings and PCs.
    [pca] >Compute explained variance.
    [pca] >Number of components is [191] that covers the [95.00%] explained variance.
    [pca] >The PCA reduction is performed on the [3072] columns of the input dataframe.
    [pca] >Fit using PCA.
    [pca] >Compute loadings and PCs.


    OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.


    
    [clusteval] >Fit using kmeans with metric: cosine, and linkage: ward
    [clusteval] >Evaluate using silhouette.


    100%|██████████| 5/5 [00:35<00:00,  7.11s/it]


    [clusteval] >Optimal number clusters detected: [40].
    [clusteval] >Fin.


# data visualization


```python
cl.results.keys()
```




    dict_keys(['img', 'feat', 'xycoord', 'pathnames', 'labels', 'url', 'filenames'])




```python
# Make various plots:

# Silhouette plots
cl.clusteval.plot()
cl.clusteval.scatter(cl.results['xycoord'])

# PCA explained variance plot
cl.pca.plot()

# Plot unique image per cluster
cl.plot_unique(img_mean=False, show_hog=True)

# Scatterplot
cl.scatter(dotsize=50, zoom=0.5, img_mean=False)

```


    
![png](output_12_0.png)
    


    [clusteval] >Estimated number of n_clusters: 39, average silhouette_score=-0.077



    
![png](output_12_2.png)
    



    
![png](output_12_3.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](output_12_5.png)
    


    [colourmap]> Warning: Colormap [Set1] can not create [40] unique colors! Available unique colors: [9].



    
![png](output_12_7.png)
    





    (<Figure size 1080x720 with 1 Axes>,
     <AxesSubplot: title={'center': 'umap plot. Samples are coloured on the cluster labels (high dimensional).'}, xlabel='x-axis', ylabel='y-axis'>)




    <Figure size 432x288 with 0 Axes>


# pictures in each cluster


```python
# Plot images per cluster or all clusters
for i in range(40):
    cl.plot(labels=i-1, show_hog=True)
```


    
![png](output_14_0.png)
    



    
![png](output_14_1.png)
    



    
![png](output_14_2.png)
    



    
![png](output_14_3.png)
    



    
![png](output_14_4.png)
    



    
![png](output_14_5.png)
    



    
![png](output_14_6.png)
    



    
![png](output_14_7.png)
    



    
![png](output_14_8.png)
    



    
![png](output_14_9.png)
    



    
![png](output_14_10.png)
    



    
![png](output_14_11.png)
    



    
![png](output_14_12.png)
    



    
![png](output_14_13.png)
    



    
![png](output_14_14.png)
    



    
![png](output_14_15.png)
    



    
![png](output_14_16.png)
    



    
![png](output_14_17.png)
    



    
![png](output_14_18.png)
    



    
![png](output_14_19.png)
    



    
![png](output_14_20.png)
    



    
![png](output_14_21.png)
    



    
![png](output_14_22.png)
    



    
![png](output_14_23.png)
    



    
![png](output_14_24.png)
    



    
![png](output_14_25.png)
    



    
![png](output_14_26.png)
    



    
![png](output_14_27.png)
    



    
![png](output_14_28.png)
    



    
![png](output_14_29.png)
    



    
![png](output_14_30.png)
    



    
![png](output_14_31.png)
    



    
![png](output_14_32.png)
    



    
![png](output_14_33.png)
    



    
![png](output_14_34.png)
    



    
![png](output_14_35.png)
    



    
![png](output_14_36.png)
    



    
![png](output_14_37.png)
    



    
![png](output_14_38.png)
    



    
![png](output_14_39.png)
    



```python

```

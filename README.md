# ADND-J7
J7 project ADND

open source code are used
1. https://www.analyticsvidhya.com/blog/2021/06/k-means-clustering-and-transfer-learning-for-image-classification/
2. https://github.com/erdogant/clustimage

conclusion until now:
1. I tried kmeans with sklean library, but the accuray is low, accuracy = 18%.
2. then I tried to extract features by pretained model resnet50, it is better, but almost very little improvement, 18.3%.
3. I tried pretained model resnet50 + hdbscan, the result is very bad.
4. I tried kmeans with clustimage library, it is much better than kmeans with sklearn lrbrary. But it seems like there are no improvment if grey pictures are used.
5. I tried agglomerative with different linkages, like "single", "complete", "centroid" and "ward". It turns out "ward" is the best.
6. all in all, clustimage library can recognize basic shapes such as circles, triangles and rectangles, but more complex shapes cannot be accurately extracted from features, and it can recognize colours such as blue, red, dark blue and black, which are more distinctive, but more similar colours cannot be recognized. 
7. **I think the key is feature extraction, as long as this step is done well, the accuracy rate will be much better later on. However, if you really want a high accuracy rate, pertained models are not enough and you need to train specifically for the gtsrb dataset.**

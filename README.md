# breast_cancer_knn_lof_pca_nca
Breast Cancer Classification Using KNN and Outlier Analysis with LOF

**Background:** Breast cancer is a type of cancer that starts in the breast. It can start in one or both breasts. Breast cancer can spread when the cancer cells get into the blood or lymph system and then are carried to other parts of the body. The lymph (or lymphatic) system is a part of your body's immune system. It is a network of lymph nodes (small, bean-sized glands), ducts or vessels, and organs that work together to collect and carry clear lymph fluid through the body tissues to the blood. The clear lymph fluid inside the lymph vessels contains tissue by-products and waste material, as well as immune system cells.

Source: [https://www.cancer.org/cancer/breast-cancer/about/what-is-breast-cancer.html]

**Motivation:** Finding breast cancer early and getting state-of-the-art cancer treatment are two of the most important strategies for preventing deaths from breast cancer. Breast cancer that’s found early, when it’s small and has not spread, is easier to treat successfully. In this work we will apply a breast cancer classification with KNN algorithm. We will also analyze the outliers of the dataset before KNN training. To increase model accuracy dimension reduction techniques like PCA and NCA will be performed.

<img src="https://studiousguy.com/wp-content/uploads/2021/08/Skewed-Distribution.jpg" width="500" height="400">

Skewness is a measurement of the distortion of symmetrical distribution or asymmetry in a data set. Skewness is demonstrated on a bell curve when data points are not distributed symmetrically to the left and right sides of the median on a bell curve. 

* A skewness value of 0 in the output denotes a symmetrical distribution of values in row 1.
* A negative skewness value in the output indicates an asymmetry in the distribution corresponding to row 2 and the tail is larger towards the left hand side of the distribution.
* A positive skewness value in the output indicates an asymmetry in the distribution corresponding to row 3 and the tail is larger towards the right hand side of the distribution.

#### Outlier Analysis
**Density based Outlier Detection: Local Outlier Factor (LOF)** : Compare local density of one point to local density of its K-NN
* LOF > 1 ==> outlier / anomaly
* LOF < 1 ==> inlier

#### K-Nearest Neighbors (KNN) Algorithm
K-nearest neighbors (KNN) is a type of supervised learning algorithm used for both regression and classification. KNN tries to predict the correct class for the test data by calculating the distance between the test data and all the training points. Then select the K number of points which is closet to the test data. The KNN algorithm calculates the probability of the test data belonging to the classes of ‘K’ training data and class holds the highest probability will be selected. In the case of regression, the value is the mean of the ‘K’ selected training points.

<img src="https://miro.medium.com/max/828/0*34SajbTO2C5Lvigs.png" width="500" height="500">

**Overfitting vs. Underfitting**

Overfitting is a concept in data science, which occurs when a statistical model fits exactly against its training data. When this happens, the algorithm unfortunately cannot perform accurately against unseen data, defeating its purpose. Generalization of a model to new data is ultimately what allows us to use machine learning algorithms every day to make predictions and classify data.

<img src="https://1.cms.s81c.com/sites/default/files/2021-03-03/model-over-fitting.png" width="600" height="250">

Source: [https://www.ibm.com/cloud/learn/overfitting]

<img src="https://1.cms.s81c.com/sites/default/files/2021-03-03/classic%20overfitting_0.jpg" width="400" height="400">

#### Dimensionality Reduction
##### Principal Component Analysis (PCA)
Principal component analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

Reducing the number of variables of a data set naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to trade a little accuracy for simplicity. Because smaller data sets are easier to explore and visualize and make analyzing data much easier and faster for machine learning algorithms without extraneous variables to process.

So, to sum up, the idea of PCA is simple — reduce the number of variables of a data set, while preserving as much information as possible.
<img alt="Principal Component Analysis second principal" src="https://builtin.com/sites/www.builtin.com/files/inline-images/national/Principal%2520Component%2520Analysis%2520second%2520principal.gif" width="700" height="400">
Source: [https://builtin.com/data-science/step-step-explanation-principal-component-analysis]

##### Neighborhood Component Analysis (NCA)
Neighbourhood components analysis is a supervised learning method for classifying multivariate data into distinct classes according to a given distance metric over the data. Functionally, it serves the same purposes as the K-nearest neighbors algorithm, and makes direct use of a related concept termed stochastic nearest neighbours.

Rather than having the user specify some arbitrary distance metric, NCA learns it by choosing a parameterized family of quadratic distance metrics, constructing a loss function of the parameters, and optimizing it with gradient descent. Furthermore, the learned distance metric can explicitly be made low-dimensional, solving test-time storage and search issues. How does NCA do this?

The goal of the learning algorithm then, is to optimize the performance of kNN on future test data. Since we don’t a priori know the test data, we can choose instead to optimize the closest thing in our toolbox: the leave-one-out (LOO) performance of the training data.

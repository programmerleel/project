from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score,silhouette_samples
import matplotlib.pyplot as plt


"""
聚类算法:无监督学习的代表
    聚类:                             分类:             
        将数据分为多个组                    从以学习好的数据学习,将数据放到对应分组
        无监督,不需要标签                   有监督,需要标签
        K-means DBSCAN 层次聚类 光谱聚类    决策树，贝叶斯，逻辑回归
        聚类结果有好有坏                    分类结果固定,u偶客观的评价指标
"""

"""
K-means:
    随机选择k个样本作为初始质心
    开始循环
    将剩余样本分配到距离最近的质心,生成k个簇
    计算每个簇所分配样本的平均值作为新的质心
    质心位置不变,停止迭代
    
聚类标准:在同一簇中的样本认为具有较高相似性,而不同簇内的样本具有较低的相似性,通过计算样本到簇的质心的距离来进行衡量
        欧氏距离
        曼哈顿距离
        余弦距离
        
        以欧氏距离举例:
            一个簇存在m个样本 一个样本存在n个特征 一个簇中所有的样本点到质心的距离平方和叫inertial
            k个簇的距离平方和叫total inertia
            total inertia越小,表示每个簇的样本越相似(类内距越小,类外差距大)
            K-means聚类的过程也就是求解total inertia最小值的过程
            
K-means聚类没有损失函数 重点!!!
    在深度学习中,需要使用损失函数进行反向传播以此来优化网络的参数,但是在K-means聚类算法中并不存在参数需要进行优化
    total inertia严格意义上并不是损失函数,而是一个评价指标的存在
    
"""

X,y = make_blobs(n_samples=10000,n_features=2,centers=4,random_state=1)

color = ["red","pink","orange","gray"]
fig, ax1 = plt.subplots(1)
for i in range(4):
    ax1.scatter(X[y==i, 0], X[y==i, 1],marker='o' ,s=8 ,c=color[i])
plt.show()

"""
评价指标
    上面提到了可以根据inertia作为K-means聚类效果的评价指标,但是存在几个缺点:
        1.极限未知,明显inertia为0时聚类效果最好,但是一个模型的inertia的极限未知
        2.特征数维度高时,计算量爆炸
        3.受簇数k的影响,inertia随着k的增大必然会减小,但这不代表模型的效果好
        4.inertia对数据分布存在凸分布假设且假设数据是各向同性
    轮廓系数:
        计算样本在与当前簇的相似度以及下一个簇的相似度
        不存在对数据的分布假设,但是在凸分布数据上会虚高
    卡林斯基-哈拉巴斯指数:
        指数越高越好

"""
n_clusters = 3

cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(X)
centers = cluster.cluster_centers_
labels = KMeans(n_clusters=n_clusters,random_state=0).fit_predict(X)
inertia = cluster.inertia_
score = silhouette_score(X,labels)
sample = silhouette_samples(X,labels)
print(inertia)
print(score)
print(sample)

color = ["red","pink","orange","gray","green","yellow"]
fig, ax1 = plt.subplots(1)
for i in range(n_clusters):
    ax1.scatter(X[labels==i, 0], X[labels==i, 1],marker='o',s=8 ,c=color[i])
ax1.scatter(centers[:,0],centers[:,1],marker="x",s=15,c="black")
plt.show()

n_clusters = 6

cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(X)
centers = cluster.cluster_centers_
labels = KMeans(n_clusters=n_clusters,random_state=0).fit_predict(X)
inertia = cluster.inertia_
score = silhouette_score(X,labels)
sample = silhouette_samples(X,labels)
print(inertia)
print(score)
print(sample)

fig, ax1 = plt.subplots(1)
for i in range(n_clusters):
    ax1.scatter(X[labels==i, 0], X[labels==i, 1],marker='o',s=8 ,c=color[i])
ax1.scatter(centers[:,0],centers[:,1],marker="x",s=15,c="black")
plt.show()


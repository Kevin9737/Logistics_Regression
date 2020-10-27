import numpy as np
import matplotlib.pyplot as plt
import imageio

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))
def h(theta,x):
    # （64，3）*（3，1）
    return sigmoid(np.dot(x,theta))
def cross_entropy_loss(y,yHat):
    # 训练集上样本的交叉熵损失
    return np.sum(-(y.T.dot(np.log(yHat))+(1-y).T.dot(np.log(1-yHat)))/yHat.shape[0])

def acc(x,y,theta):
    # 测试集上的判定准确度
    pre=x.dot(theta) >0
    pre = np.int64(pre)
    # 预测标签序列pre与真实标签序列比较
    return np.sum(np.sum(pre[i]==y[i] for i in range(len(y)))/len(y))

def logistics_regression(alpha, maxloop, X, Y):
    # 随机初始化参数
    theta = np.random.randn(3,1)
    count = 0  # 记录迭代次数
    theta_list=[]
    loss_list=[]
    acc_list=[]
    image_list=[]
    while count <= maxloop:
        # 参数更新，一般除以样本点个数
        theta=theta+1.0*alpha*np.dot(X.T,(Y-h(theta, X)))/X.shape[0]
        # theta = theta + 1.0 * alpha * np.dot(X.T, (Y - h(theta, X)))
        yHat=X.dot(theta)
        yHat=sigmoid(yHat)
        # print(yHat)

        loss_list.append(cross_entropy_loss(Y,yHat))
        acc_list.append(acc(test_X,test_Y,theta))
        theta_list.append(theta)
        count+=1
    return theta,theta_list,loss_list,acc_list
    pass



if __name__ == '__main__':
    alpha = 0.6 # 学习率learning rate
    maxloop = 1000 # 最大迭代次数
    global X,Y,test_X,test_Y
    X = np.loadtxt(r'exam\train\x.txt')#最普通的loadtxt
    Y = np.loadtxt(r'exam\train\y.txt')
    test_X = np.loadtxt(r'exam\test\x.txt')#最普通的loadtxt
    test_Y = np.loadtxt(r'exam\test\y.txt')

    #将Y由(64,)->(64,1),即在Y的原先维度上增加一个维度，使之变成列向量
    Y=np.expand_dims(Y,axis=1)

    #归一化训练集
    # axis=0表示行压缩，即求一列上的最小值
    X = X - np.min(X, axis=0, keepdims=True)
    X = X / (np.max(X, axis=0, keepdims=True) - np.min(X, axis=0, keepdims=True))
    # 归一化测试集
    test_X = test_X - np.min(test_X, axis=0, keepdims=True)
    test_X = test_X / (np.max(test_X, axis=0, keepdims=True) - np.min(test_X, axis=0, keepdims=True))

    # 将X加上一列1（偏置）
    # axis=0 拼横着的数（纵向），axis=1拼纵列的数（横向）
    # 将第一列为1的列向量，与原X（test_X）矩阵相连
    X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
    test_X = np.concatenate((np.ones((test_X.shape[0],1)), test_X), axis=1)

    theta_list,loss_list,acc_list,image_list=[],[],[],[]
    # 得到迭代更新后的theta,loss变化情况列表loss_list
    theta,theta_list,loss_list,acc_list=logistics_regression(alpha,maxloop,X,Y)
    # print("="*100)
    # print("loss_list:",loss_list)
    # print("acc_list:",acc_list)
    # print("="*100)

    # 得到一行三张子图的fig（画布），ax（图表）对象
    fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(16,9))
    #每次循环迭代过程：
    # ax[0]先画点，再画线，由于下次的classification line位置会发生偏移，所以画完之后得清除旧图ax[0]，即plt.cla()
    # ax[1]和ax[2]的画法有两种：
    # （1）：每次迭代画一个新的loss或者acc点上去，不需要清楚旧图
    # （2）：每次迭代将更新后的loss_list或者acc_list列表上面的所有点都画上去，此时在画完之后下次循环之前是否擦除这个旧图，视觉效果上，结果都是一样的
    # 以下ax[1]和ax[2]采用方案（2）
    for i in range(maxloop):
        # s：控制散点的大小
        # c：控制散点颜色
        # marker：控制散点样式
        # label：表示散点名称
        # ax[0].set_ylim([0,1])
        ax[0].scatter(X[:,1][Y.flatten() == 1], X[:,2][Y.flatten() == 1], s=40, c='b', marker='*', label='admitted') #画出正类样本=点
        ax[0].scatter(X[:,1][Y.flatten() == 0], X[:,2][Y.flatten() == 0], s=40, c='r', marker='o', label='not admitted') #画出负类样本点
        ax[0].set_title('LogisticRegression_GD')
        line_x = np.arange(0, 1, 0.1)
        # print("theta_list[i]:",theta_list[i])
        line_y = -theta_list[i][0, 0] / theta_list[i][2, 0] - (theta_list[i][1, 0] / theta_list[i][2, 0]) * line_x
        # 描点画线。c='g'表示画线
        ax[0].plot(line_x,line_y, c='g', label='boundary')
        # 显示图例
        ax[0].legend()

        # 设置标题以及x，y轴的名称
        ax[1].set_title('loss change')
        ax[1].set_xlabel('iterations')
        ax[1].set_ylabel('loss')
        # 'b.'表示画蓝点，c='b'表示画线
        # 图中画i个点，且横轴以1为起点
        print("current_loss:",loss_list[i])
        ax[1].plot(np.array(range(1,i+1)), np.array(loss_list[:i]), 'b.')

        ax[2].set_title('accuracy change')
        ax[2].set_xlabel('iterations')
        ax[2].set_ylabel('accuracy')
        ax[2].plot(np.array(range(1,i+1)), np.array(acc_list[:i]), 'r.')
        # pause函数可以替代plt.show()展示所画的图形，并且停顿0.1s
        # 两者不同的地方在于，pause函数是交互模式，不影响程序的继续运行。
        # plt.show()函数展示图片之后，程序停止继续运行，只有关了图片之后才会继续往后运行
        plt.pause(0.1)
        # 每得到10张图片便保存一张
        if i % 10 == 0:
            plt.savefig('gd.png')
            image_list.append(imageio.imread('gd.png'))
        # 因为classification line会发生变化，所以第一张子图的旧图必须清理
        # 第二张第三张子图任意
        ax[0].cla()
    # 保存为gif图
    imageio.mimsave('gd.gif', image_list, duration=0.1)


'''
1. 訓練準備
	1.1. 通過pytorch取得mnsit資料
	1.2. 建立網路
	1.3. 建立批節點
2. 訓練流程
	2.1. 設定池和優化器
	2.2. 開始訓練
	2.3. 測試在測試資料集上的表現
3. 保存模型
	3.1. 移除批節點
	3.2. 保存
'''

######################### 1. 訓練準備 ##############################

# 1.1. 通過pytorch取得mnsit資料

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset=datasets.MNIST('./mnist', train=True, transform=transform, download=True)
test_dataset=datasets.MNIST('./mnist', train=False, transform=transform, download=True)

train_loader=DataLoader(train_dataset, batch_size=len(train_dataset))
test_loader=DataLoader(test_dataset, batch_size=len(test_dataset))

train_img=next(iter(train_loader))[0].numpy()
train_label=next(iter(train_loader))[1].numpy()
train_label=np.eye(10)[train_label]

test_img=next(iter(test_loader))[0].numpy()
test_label=next(iter(test_loader))[1].numpy()
test_label=np.eye(10)[test_label]

# 1.2. 建立網路

from nyto import net_tool as to
from nyto import layer
from nyto import unit_function as uf

(nn,n)=to.new_net(
    conv1=layer.new_conv_layer((1,6), pad_mod='same'),
    conv2=layer.new_conv_layer((6,36), pad_mod='same', kernal_size=(5,5)),
    
    nn1=layer.new_nn_layer((324, 160)),
    nn2=layer.new_nn_layer((160, 10))
)

n.conv_layer1=n.data_input>>n.conv1>>uf.tanh()>>uf.max_pooling((3,3),2)
n.conv_layer2=n.conv_layer1>>n.conv2>>uf.tanh()>>uf.max_pooling((5,5),3)
n.flat=n.conv_layer2>>uf.flattening()

n.pre=n.flat>>n.nn1>>uf.tanh()>>n.nn2>>uf.softmax()
n.loss=uf.cross_entropy(n.pre, n.data_label)
n.accuracy=uf.accuracy(n.pre, n.data_label)

# 1.3. 建立批節點

nlist=to.batch_launcher(
    nn=nn,
    get={n.pre,n.loss,n.accuracy},
    batch_push={'data_input':train_img, 'data_label':train_label},
    batch_size=100
)

loss_nlist=[node['loss'] for node in nlist]
acc_nlist=[node['accuracy'] for node in nlist]

train_loss_nlist=loss_nlist[:-1]
train_acc_nlist=acc_nlist[:-1]

val_loss_n=loss_nlist[-1]
val_acc_n=acc_nlist[-1]

######################### 2. 訓練流程 ##############################

# 2.1. 設定池和優化器

from nyto import train_tool as train

pool=train.new_pool(
    loss_nlist[0],
	pool_size=6,
	random_size=0.2,
	keep=True
)

opt=train.epso_opt(
    threshold=1,
	step_rate=0.1,
	bound=100
)

# 2.2. 開始訓練

train_iter=train.train_batch(
    train_loss_nlist, pool, opt, step=5, epoch=1
)

for t, new_pool, opt in train_iter:
    (epoch, step)=t
    
    best_net=new_pool.net[0]
    best_loss=new_pool.loss[0]
    
    train_acc_n=train_acc_nlist[step]
    best_acc=toget(best_net[train_acc_n.node_id])
    
    (val_loss,val_acc)=toget(
        best_net[val_loss_n.node_id], best_net[val_acc_n.node_id]
    )
    
    print(f"t:{step} train: loss={best_loss:.4f} acc={best_acc} val: loss={val_loss:.4f} acc={val_acc}")
	
# 2.3. 測試在測試資料集上的表現

test_performance=best_net.push_get(
    {'loss', 'accuracy'},
    data_input=test_img[:100],
    data_label=test_label[:100],
)

print(f"test: loss={test_performance['loss']} acc={test_performance['accuracy']}")

######################### 3. 保存模型 ##############################

# 3.1. 移除批節點

remove_id_set=to.free_unused_nodes(best_net, {'loss', 'accuracy'})

# 3.2. 保存

import pickle

with open('my_net.pickle', 'wb') as f:
    pickle.dump(best_net, f)
	
'''
# 恢復模型
with open('my_net.pickle', 'rb') as f:
    my_net=pickle.load(f)
'''

############################ END #################################
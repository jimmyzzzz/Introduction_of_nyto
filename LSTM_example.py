'''
1. 訓練準備
	1.1. 取得並整理0050股價資料
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

# 1.1. 取得0050股價資料

import pandas as pd
import numpy as np

def get_feature_label_list(data_df):
	feature_cols=['volume','open','high','low','close','record']
	label_col='label'
	label_n=2

	feature_list=[]
	label_list=[]
	for stock_id, df in data_df.groupby(by='stock_id'):
		df=df.sort_values(by=['date'])
		feature_df=df[feature_cols].replace(np.inf, 0)
		label_df=df[label_col]

		feature_np=feature_df.values
		label_np=np.eye(label_n)[label_df.values]

		feature_list.append(feature_np)
		label_list.append(label_np)

	return feature_list, label_list

df=pd.read_csv('data/s0050_data.csv')
(feature_list,label_list)=get_feature_label_list(df)

# 1.2. 建立網路

from nyto import net_tool as to
from nyto.net_tool import get as toget
from nyto import layer
from nyto import unit_function as uf

def encode_to_np(list_np):
	(splite_list, tmp)=([], 0)
	for n in list_np:
		tmp+=len(n)
		splite_list.append(tmp)

	return np.vstack(list_np), splite_list[:-1]

def decode_to_list(total_np, splite_list):
	return np.split(total_np, splite_list, axis=0)

def encode_then_decode(*mods, data_list):
	(data_np, splite_list)=encode_to_np(data_list)

	return_np=data_np
	for mod in mods: return_np=mod(return_np)

	return decode_to_list(return_np, splite_list)

def loss_function(pre_list, label_list, miss_n=25, test_n=100, val_n=100):
	(cat_pre_list, cat_label_list)=([], [])
	for pre_np, label_np in zip(pre_list, label_list):
		cat_pre_list.append

	test_idx=-test_n
	val_idx=-(test_n+val_n)

	test_pre_np=np.vstack([pre_np[test_idx:] for pre_np in pre_list])
	test_label_np=np.vstack([label_np[test_idx:] for label_np in label_list])

	val_pre_np=np.vstack([pre_np[val_idx:test_idx] for pre_np in pre_list])
	val_label_np=np.vstack([label_np[val_idx:test_idx] for label_np in label_list])

	train_pre_np=np.vstack([pre_np[miss_n:val_idx] for pre_np in pre_list])
	train_label_np=np.vstack([label_np[miss_n:val_idx] for label_np in label_list])

	test_loss=uf._cross_entropy(test_pre_np, test_label_np)
	test_acc=uf._accuracy(test_pre_np, test_label_np)

	val_loss=uf._cross_entropy(val_pre_np, val_label_np)
	val_acc=uf._accuracy(val_pre_np, val_label_np)

	train_loss=uf._cross_entropy(train_pre_np, train_label_np)
	train_acc=uf._accuracy(train_pre_np, train_label_np)

	return train_loss,train_acc,val_loss,val_acc,test_loss,test_acc

(nn,n)=to.new_net(
	proc=to.add_data(encode_then_decode),
	loss_function=to.add_data(loss_function),

	nn1=layer.new_nn_layer((6,24)),
	lstm1=layer.new_lstm_layer((24,12)),
	lstm2=layer.new_lstm_layer((12,2))
)

n.nn_layer=n.proc(n.nn1, uf._relu, data_list=n.data_input)
n.lstm_layer=n.nn_layer>>n.lstm1>>n.lstm2
n.pre=n.proc(uf._softmax, data_list=n.lstm_layer)

n.performance=n.loss_function(n.pre, n.data_label)
n.loss=n.performance[0]

# 1.3. 建立批節點

nlist=to.batch_launcher(
	nn=nn,
	get={n.performance, n.loss, n.data_input, n.data_label},
	batch_push={
		'data_input':feature_list,
		'data_label':label_list
	},
	batch_size=2
)

loss_nlist=[n['loss'] for n in nlist]

######################### 2. 訓練流程 ##############################

# 2.1. 設定池和優化器

from nyto import train_tool as train

pool=train.new_pool(
	loss_nlist[0], pool_size=6, random_size=0.2, keep=True
)

opt=train.epso_opt(
	threshold=1, step_rate=0.1, bound=100
)

# 2.2. 開始訓練

train_iter=train.train_batch(
	loss_nlist, pool, opt, step=1, epoch=1
)

for t, new_pool, opt in train_iter:
	(epoch, step)=t

	best_net=new_pool.net[0]
	best_loss=new_pool.loss[0]

	performance_n=nlist[step]
	performance=toget(best_net[performance_n.node_id])['performance']

	(train_loss,train_acc,val_loss,val_acc,test_loss,test_acc)=performance

	print(f"t:{step} train: loss={train_loss:.4f} acc={train_acc} val: loss={val_loss:.4f} acc={val_acc}")

# 2.3. 測試在測試資料集上的表現

test_performance=best_net.push_get(
	{'performance'},
	data_input=feature_list,
	data_label=label_list,
)['performance']

print(f"test: loss={test_performance[4]} acc={test_performance[5]}")

######################### 3. 保存模型 ##############################

# 3.1. 移除批節點

remove_id_set=to.free_unused_nodes(best_net, {'performance', 'loss'})

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
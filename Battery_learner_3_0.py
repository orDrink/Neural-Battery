import datetime
import random
import re
import csv
import logging, os
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import mixed_precision
from typing import List, Union, Optional

class DynamicProcessSolver(tf.keras.Model):
    
    def __init__(self, unit, num_outputs, **kwargs):
        
        super().__init__(**kwargs)
        self.unit = unit
        self.num_outputs = num_outputs
        self.mean = None
        self.variance = None
        self._build_layers()     

    def _build_layers(self):
        
        self.norm = tf.keras.layers.Normalization(axis=-1, mean=self.mean, variance=self.variance)     
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.unit*2, activation='tanh'),
            tf.keras.layers.Dense(self.unit, activation=None,kernel_regularizer=regularizers.l2(0.01))  # 线性激活
        ])  
        self.update = tf.keras.Sequential([
            tf.keras.layers.Dense(self.unit)
        ])  
        self.layer_norm2 = tf.keras.layers.LayerNormalization()
        self.beta = tf.keras.layers.Dense(1,activation=None, #ect
                                          use_bias=False,
                                          kernel_initializer='glorot_uniform')
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(self.unit*2, activation='tanh'),
            tf.keras.layers.Dense(self.num_outputs, activation=None)  # 输出维度与输入一致
        ])

    def _process_step(self, x, states):
                 
        x = self.layer_norm1(x)
        x = self.flatten(x)
        x = self.encoder(x)
        states = self.update(x)  
        states = self.layer_norm2(states)
        ect = self.beta(x)       
        combined = tf.concat([states,ect], axis=-1)            
        y = self.decoder(combined)
        
        return y, states # 返回张量 y    
    
    
    def call(self, inputs, lookback = 3, fix = [], state_sequence = False):
        
        # 输入形状: (batch_size,seq len, input_dim)
        
        # 初始化
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]       
        fix = tf.convert_to_tensor(fix, dtype=tf.int32)
        num_fix = tf.shape(fix)[0]
        
        initial_tensor = inputs[:,:,1:]
        loading_tensor = tf.expand_dims(inputs[:,:,0], axis=-1) 
        states = tf.zeros((batch_size, self.unit))
        outputs = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        state_sequence_buffer = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        window_buffer = tf.zeros_like(initial_tensor[:, :lookback, :])
                               
        # 计算填充积分列和微分列     
        integral = tf.cumsum(loading_tensor, axis=-1)
        diff = loading_tensor[:, 1:, :] - loading_tensor[:, :-1, :]
        diff = tf.pad(diff, paddings=[[0,0], [1,0], [0,0]])
        loading_tensor_updated = tf.concat([loading_tensor, integral, diff], axis=-1)
#         loading_tensor_updated = loading_tensor
                                    
        # 对约束输出进行修正的参数（如果指定的话）
        indices = tf.stack([
            tf.repeat(tf.range(batch_size), num_fix),
            tf.tile(fix, [batch_size])
        ], axis=1)                  
        updates = tf.gather(initial_tensor[:,1:seq_len,:], fix, axis=-1)       
        
        # 根据时间窗口进行预数据加载 
        _, outputs = tf.while_loop(
            cond=lambda j, _: j < lookback,
            body=lambda j, arr: (j + 1, arr.write(j, initial_tensor[:, j, :])),
            loop_vars=(0, outputs),
            parallel_iterations=4  # 提高并行度
        )        
        window_buffer = tf.transpose(outputs.stack()[-lookback:], [1, 0, 2])      
        def body(i, states, outputs, window_buffer, state_sequence_buffer):
                            
            # 组装输入
            loading_conditions = loading_tensor_updated[:, i-lookback:i,:]  # shape: (batch_size, seq_len, 3)  
            x = tf.concat([loading_conditions[:,-lookback:,:],window_buffer], axis=-1)            
            
            # 迭代步
            y, new_states = self._process_step(x, states)  # y 是张量
     
            # 对约束输出进行修正   
            y = tf.tensor_scatter_nd_update(y, indices, tf.reshape(updates[:,i,:], [-1]))   
            
            # 写输出     
            outputs = outputs.write(i, y)  # shape: (seq_len, batch_size, num_outputs)
            window_buffer = tf.concat([window_buffer[:, 1:, :], y[:, tf.newaxis, :]], axis=1)
            state_sequence_buffer = state_sequence_buffer.write(i, new_states)
            return i + 1, new_states, outputs, window_buffer, state_sequence_buffer

        _, final_state, final_outputs,_ , state_sequence_buffer= tf.while_loop(
            cond=lambda i, *_: i < seq_len - 1,
            body=body,
            loop_vars=(lookback, states, outputs, window_buffer, state_sequence_buffer),
            parallel_iterations=1
#            swap_memory=True
        )

        stacked = final_outputs.stack()  # shape: (seq_len, batch_size, num_outputs)
        final_output = tf.transpose(stacked, [2, 1, 0])  # shape: (num_outputs, batch_size, seq_len)
        final_output = final_output[:,:,lookback:]
        
        # 返回逻辑，状态序列选项
        if state_sequence:
            return [final_output[i] for i in range(self.num_outputs)], state_sequence_buffer.stack()
        else:
            return [final_output[i] for i in range(self.num_outputs)]       
   
    def get_config(self):
        config = super().get_config()
        config.update(
            {"unit": self.unit,
             "num_outputs": self.num_outputs,
            }
        )
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config) 
    
def sample_reader(file_name: str, columns: [str], lookback: int = 1) :
    """数据读取"""
    try:
        df = pd.read_csv(f"{file_name}.csv", usecols=columns)
        data = df[columns].values.astype(np.float32)
        
        # 生成滑动窗口
        samples = []
        for i in range(len(data) - lookback + 1):
            samples.append(data[i:i+lookback])
            
        return np.array(samples)
    except Exception as e:
        print(f"读取 {file_name} 失败: {str(e)}")
        return np.empty((0, lookback, len(columns)))
    
def series_generation(data: np.ndarray, num_steps: int = 20) :
    """序列生成"""
    if len(data) == 0:
        return np.empty((0, num_steps, data.shape[1], data.shape[2]))
    
    series = []
    for i in range(len(data) - num_steps + 1):
        series.append(data[i:i+num_steps])
        
    return np.array(series)

def preprocess(
    filelist: [str],
    columns: [str] = None,
    lookback: int = 3,
    num_steps: int = 20,
    max_samples: int = 6000
) :
    """预处理流程"""
    dataset = []
    
    for file in filelist:
        try:
            # 读取数据
            samples = sample_reader(file, columns, lookback)
            if len(samples) == 0:
                continue
                
            # 生成序列
            series = series_generation(samples, num_steps)
            if len(series) == 0:
                continue
                
            # 随机采样
            if len(series) > max_samples:
                indices = np.random.choice(len(series), max_samples, replace=False)
                series = series[indices]
              
            dataset.append(series)

        except Exception as e:
            print(f"处理 {file} 时出错: {str(e)}")
            continue
    
    # 合并所有数据
    dataset = np.concatenate(dataset, axis=0) 
    dataset = dataset.reshape(-1, num_steps, len(columns))
    
    print(dataset.shape)
    return dataset.astype(np.float32)

def data_shuffle_split(dataset, test_size=0.1):

    idx = np.random.permutation(len(dataset))  # 生成随机索引
    dataset = dataset[idx]  # 打乱
    split = int(len(dataset) * (1 - test_size))  # 计算分割点
    return dataset[:split], dataset[split:]

class ConservativeLoss(tf.keras.losses.Loss):
    
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, lamda = 0.5, name='conservative_loss'):
        super().__init__(reduction=reduction, name=name)
        self.lamda = tf.constant(lamda, dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        
        y_invariant = y_pred[:,-1]    
        y_main = y_pred[:,:-1]       
        y_true = tf.cast(y_true[:,:-1], tf.float32)  # 类型转换    
        loss1 = tf.losses.mean_squared_error(y_true, y_main)      
        ones_ref = tf.ones(tf.shape(y_invariant))
        loss2 = tf.losses.mean_squared_error(ones_ref,y_invariant)       
        loss = (1 - self.lamda)*loss1 + self.lamda*loss2 
        
        return tf.reduce_mean(loss)

class MultiTaskAdEMAMix(Optimizer):
    def __init__(self,
                 learning_rate=0.001,
                 beta_1=0.9,
                 beta_2=0.999,
                 beta_3=0.99,
                 task_config=None,  # 格式: {task_id: {"alpha": 0.2, "beta_3": 0.95}, ...}
                 task_regex=r'task_(\d+)',  # 从变量名提取任务ID的正则
                 epsilon=1e-7,
                 name="MultiTaskAdEMAMix",
                 **kwargs):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.task_regex = task_regex
        self.epsilon = epsilon
        
        # 使用 _set_hyper 注册超参数
        self._set_hyper("learning_rate", kwargs.get('learning_rate', learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        
        # 初始化任务配置
        self.task_config = task_config if task_config else {0: {"alpha": 0.2, "beta_3": beta_3}}
        for task_id in self.task_config:
            self._set_hyper(f"beta_3_{task_id}", self.task_config[task_id]["beta_3"])
            self._set_hyper(f"alpha_{task_id}", self.task_config[task_id]["alpha"])

    def _create_slots(self, var_list):
        pattern = re.compile(self.task_regex)
        for var in var_list:
            var_name = var.name
            match = pattern.search(var_name)
            task_id = int(match.group(1)) if match else 0
            if task_id not in self.task_config:
                task_id = 0  # 默认任务
            
            # 为每个变量创建对应任务的slot
            self.add_slot(var, f'm_{task_id}') 
            self.add_slot(var, f'v_{task_id}')
            self.add_slot(var, f'ema_fast_{task_id}')
            self.add_slot(var, f'ema_slow_{task_id}')

    def _resource_apply_dense(self, grad, var, apply_state=None):
        
        # 从变量名解析任务ID
        var_name = var.name
        match = re.search(self.task_regex, var_name)
        task_id = int(match.group(1)) if match else 0
        if task_id not in self.task_config:
            task_id = 0
        
        # 获取任务特定参数
        beta_1 = self._get_hyper("beta_1", tf.float32)
        beta_2 = self._get_hyper("beta_2", tf.float32)
        beta_3 = self._get_hyper(f"beta_3_{task_id}", tf.float32)
        alpha = self._get_hyper(f"alpha_{task_id}", tf.float32)
        lr = self._get_hyper("learning_rate", tf.float32)
        epsilon = self.epsilon

        # 获取对应任务的slot
        m = self.get_slot(var, f'm_{task_id}')
        v = self.get_slot(var, f'v_{task_id}')
        ema_fast = self.get_slot(var, f'ema_fast_{task_id}')
        ema_slow = self.get_slot(var, f'ema_slow_{task_id}')

        # Adam参数更新
        m_t = m.assign(beta_1 * m + (1 - beta_1) * grad)
        v_t = v.assign(beta_2 * v + (1 - beta_2) * tf.square(grad))
        m_hat = m_t / (1 - tf.pow(beta_1, tf.cast(self.iterations + 1, tf.float32)))
        v_hat = v_t / (1 - tf.pow(beta_2, tf.cast(self.iterations + 1, tf.float32)))

        # 任务特定的EMA更新
        ema_fast_t = ema_fast.assign(beta_3 * ema_fast + (1 - beta_3) * grad)
        ema_slow_t = ema_slow.assign(beta_1 * ema_slow + (1 - beta_1) * grad)
        
        # 动态alpha调度（示例：线性增长）
        alpha_current = alpha * (1 - tf.exp(-tf.cast(self.iterations, tf.float32) / 1000))
        mixed_grad = alpha_current * ema_slow_t + (1 - alpha_current) * ema_fast_t

        # 应用更新
        var_update = var - lr * mixed_grad / (tf.sqrt(v_hat) + epsilon)
        return var.assign(var_update)

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "beta_1": self._serialize_hyperparameter("beta_1"),
            "beta_2": self._serialize_hyperparameter("beta_2"),
            "task_config": {
                task_id: {
                    # 调用value()方法获取数值
                    "alpha": float(self._hyper[f"alpha_{task_id}"].value()),
                    "beta_3": float(self._hyper[f"beta_3_{task_id}"].value())
                } for task_id in self.task_config
            },
            "task_regex": self.task_regex,
            "epsilon": self.epsilon
        })
        return config

    @classmethod
    def from_config(cls, config):
        # 确保task_config的键为整数
        task_config = {
            int(task_id): params 
            for task_id, params in config.pop("task_config").items()
        }
        return cls(task_config=task_config, **config) 

def data_shuffle(inputs = None, outputs = None, seed = 1):

    np.random.seed = seed
    idx = np.random.permutation(len(outputs))
    inputs_shuffle = inputs[idx]
    outputs_shuffle = outputs[idx]

    return inputs_shuffle, outputs_shuffle

def training(dataset =[],
             batch_size = 256, validation_split = 0.1,
             learner = None, units = 4,lookback=3,
             optimizer = None,
             loss = None, min_lr=0.0001,
             n_inv = 0,
             epochs = 200,
             root_url = "",
             name = "unknown",
             mixed_pre = False,
             loss_weights= None
            ):
    
    n_outputs = dataset.shape[-1]-1+n_inv
    
    if n_inv>0:
        ones_to_add = tf.ones([
            tf.shape(dataset)[0],  
            tf.shape(dataset)[1],
            n_inv
        ])
        dataset = tf.concat([dataset, ones_to_add], axis=-1)  
    
    model = learner(units,n_outputs)

    if  mixed_pre:
        policy = mixed_precision.Policy('mixed_float16')
        model._set_dtype_policy(policy)

    input_initial = tf.constant(dataset[:2])
 
    y = model(input_initial,lookback = lookback)
    print(input_initial.shape)
    print(len(y))
    print(y[0].shape)
    model.summary()

    callbacks = [
      keras.callbacks.ModelCheckpoint(
          'best.tf', save_best_only=True, monitor="val_loss", mode='min', save_weights_only=True
      ),
      keras.callbacks.ReduceLROnPlateau(
          monitor="val_loss", factor=0.5, patience=10, min_lr=min_lr, verbose=1
      ),
      keras.callbacks.EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=1),
      keras.callbacks.TerminateOnNaN(),
    ]
    
    model.compile(       
        optimizer = optimizer,
        loss = loss, #'mse'  
        loss_weights = loss_weights
#        loss_weights= {f'output_{i+1}': 1/n_outputs  for i in range(n_outputs)},
    )    
            
    history = model.fit(       
        x=dataset,
        y=[dataset[:,lookback:-1,i+1] for i in range(n_outputs)],
        batch_size = batch_size,
        validation_split = validation_split,      
        epochs=epochs,
        callbacks=callbacks,
    )

    save_path = os.path.join(root_url, name)
    model.save(save_path, save_format='tf')
    
    return history


def plot_histroy(history = None, n_outputs = 1, root_url = '',name = '',columns = []):

    metrics = [f'output_{i+1}_loss' for i in range(n_outputs)]    
    historyfile = open(root_url+name+".csv","w", encoding='utf-8', newline='')
    writer = csv.writer(historyfile)
    
    for step in range(len(history.history[metrics[0]])):      
        line = [step]     
        for k in range(n_outputs):          
            line.append(history.history[metrics[k]][step])
            line.append(history.history["val_" + metrics[k]][step])        
        writer.writerow(line)
     
    plot_params = [
        (history.history[metric], history.history["val_" + metric]) 
        for metric in metrics
    ]     

    fig, axs = plot_comparisons(data_pairs = plot_params, 
            labels = ["train", "val"],
            xlabel = "epoch",
            ylabel = ["loss"] * len(columns[1:]),
            titles = columns[1:],
            font_family = 'Times New Roman',
            font_size = 16,
            figsize = (12, 12),
            share_axis = False,
            axs_type = 'log')
       
    fig.savefig(
        root_url+name+".png",  # 文件名（支持PNG/JPEG/PDF等格式）
        dpi=300,        # 分辨率（默认100）
        bbox_inches="tight",  # 裁剪空白边缘
        transparent=True       # 透明背景（可选）
    )
                
# 模型验证部分          
def plot_comparisons(data_pairs: List[tuple], 
                    labels: List[str], 
                    ylabel: str,
                    xlabel: str = "Time-step",
                    titles: Optional[List[str]] = None,
                    font_family: str = 'sans-serif',
                    font_size: int = 12,
                    figsize: tuple = (12, 8),
                    subplot_layout: tuple = None,
                    share_axis: bool = True,
                    axs_type: str = 'linear'):
    """
    多对数据对比绘图函数
    
    Parameters:
        data_pairs   : 包含多对(predictions, references)的列表
        labels       : 图例标签 [预测标签, 参考标签]
        ylabel       : Y轴标签
        titles       : 每个子图的标题列表 (可选)
        font_family  : 字体类型 (默认'sans-serif')
        font_size    : 字号 (默认12)
        figsize      : 图像尺寸 (默认(12,8))
        subplot_layout : 子图布局 (nrows, ncols)，默认自动计算
        share_axis   : 是否共享坐标轴 (默认True)
    """
    # 设置全局字体
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size
    
    n_plots = len(data_pairs)
    
    # 自动计算子图布局
    if not subplot_layout:
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        subplot_layout = (rows, cols)
    
    # 创建子图网格
    fig, axs = plt.subplots(*subplot_layout, 
                           figsize=figsize, 
                           sharex=share_axis, 
                           sharey=share_axis)
    axs = np.array(axs).flatten()  # 统一处理一维数组
    
    # 遍历数据对绘制子图
    for idx, (ax, (pred, ref)) in enumerate(zip(axs, data_pairs)):
            
        if axs_type == 'linear':            
            ax.plot(pred, label=labels[0], lw=1.5, alpha=0.8)
            ax.plot(ref, label=labels[1], ls='--', alpha=0.8)
            
        if axs_type == 'log': 
            ax.semilogy(pred, label=labels[0], lw=1.5, alpha=0.8)
            ax.semilogy(ref, label=labels[1], ls='--', alpha=0.8)
            
        ax.set_ylabel(ylabel[idx])
        ax.set_xlabel(xlabel)
        ax.legend(loc='upper right', frameon=False)
        
        # 设置子图标题
        if titles and idx < len(titles):
            ax.set_title(titles[idx], pad=10)
        
        # 关闭多余子图
        if idx >= n_plots - 1:
            for ax in axs[idx+1:]:
                ax.set_visible(False)
            break
    
    plt.tight_layout()
    return fig, axs
    
def load_tf_model(model_path, custom_objects):
    """封装模型加载逻辑"""
    return keras.models.load_model(
        model_path,
        custom_objects=custom_objects
    )
             
def prepare_data(dataset, split_params):
    """数据预处理统一入口"""
    dataset = np.array(dataset).reshape(len(dataset), -1).astype(np.float32)
    if split_params.get('split'):
        start = split_params['start']
        return dataset[start:start+split_params['num_predict']+1]
    return dataset

def plot_comparison(predictions, references, labels, ylabel):
    """通用对比绘图函数 分图"""
    
#     print(predictions.shape)
#     print(references.shape)
  
    plt.figure()
    plt.plot(predictions, label=labels[0])
    plt.plot(references, label=labels[1])
    plt.ylabel(ylabel, fontsize="large")
    plt.xlabel("time-step", fontsize="large")
    plt.legend(loc="best")
    plt.show()
    plt.close()

def model_tester(
    model = None,
    model_name='',
    file_name='',
    columns=[],
    title=[],
    lookback=3,
    n_inv = 0,
    fix = [],
    split_params={'split': False, 'start': 1, 'num_predict': 50},
    mean=None,
    variance=None
):
    
    # 数据准备
    raw_data = sample_reader(
        file_name=file_name,
        columns=columns,
        lookback=1
    )
    
    processed_data = prepare_data(raw_data, split_params)    
    
    if n_inv > 0:
        ones_to_add = tf.ones([
            tf.shape(processed_data)[0],  
            n_inv])
        processed_data = tf.concat([processed_data, ones_to_add], axis=-1)
        columns += ["inv"] * n_inv
    
    # 模型加载
    model = load_tf_model(
        f"{model_name}",
        { model.__name__: model,
        'MultiTaskAdEMAMix': MultiTaskAdEMAMix,
        'ConservativeLoss': ConservativeLoss}
    )
    
    # 模型预测
    inputs = tf.expand_dims(processed_data, axis=0)
    predictions = model(inputs, lookback = lookback, fix=fix)
    
    # 预测结果保存
    predictionfile = open(model_name+"+"+file_name+".csv","w", encoding='utf-8', newline='')
    writer = csv.writer(predictionfile)  
    n_outputs = len(predictions)
    line = ['time']
    line.extend(columns[1:])
    writer.writerow(line)
    for step in range(len(predictions[0][0])):          
        line = [step]     
        for k in range(n_outputs):          
            line.append(float(predictions[k][0][step]))      
        writer.writerow(line)
           
    # 可视化对比     
    plot_params = [
        (predictions[i][0,:], processed_data[1:, i + 1]) 
        for i in range(len(predictions))
    ]
       
    plot_comparisons(data_pairs = plot_params, 
                labels = ["Pred", "True"], 
                ylabel = title,
                titles = None,
                font_family = 'Times New Roman',
                font_size = 16,
                figsize = (12, 12),
                share_axis = False)

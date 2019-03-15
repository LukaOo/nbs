import matplotlib.pyplot as plt
import torch
import PIL
import pickle

def test_conv(conv_image, test_image_path, bin_test=True):
    """
    Tests convolution results
    Args:
    conv_image: tensor: result tensor of convolved image
    test_image_path: str: path to true tensor saved before 
    """
    o_image = conv_image    
    t_image = torch.load(test_image_path) # тут проверочный выход читанем его 
    print(o_image.size())
    assert o_image.size() == t_image.size(), 'Размер выходного изображения не совпадает с размером теста'
    # смотрим, что получилось
    fig=plt.figure(figsize=(15, 15))
    p = fig.add_subplot(1, 2, 1)
    p.set_title('Результат свертки')
    plt.imshow(o_image.numpy(), cmap='gray')
    p = fig.add_subplot(1, 2, 2)
    p.set_title('Тест')
    plt.imshow(t_image.numpy(), cmap='gray')
    if bin_test:
        assert (o_image == t_image).all(), 'Что-то пошло не так, тест не пройден'

def test_bn_plots(mean, var, stat_name):
    t_mean = None
    t_var = None
    with open ('./data/'+stat_name+".npy", 'rb') as fp:
        t_mean, t_var = pickle.load(fp)        
    plt.figure(figsize=(8,8))
    ax = plt.subplot(1,2,1)
    ax.plot(t_mean,label="mean_"+stat_name + ' test')
    ax.plot(mean,label="mean_"+stat_name + ' your')
    ax.legend()
    ax = plt.subplot(1,2,2)
    ax.plot(t_var,label="var_"+stat_name + ' test')
    ax.plot(var,label="var_"+stat_name + ' your')
    ax.legend()

import os,sys  
from torchsummary import summary 
from torchviz import make_dot
from torchview import draw_graph
import torch
from torchinfo import summary as info



def Visualize(model,x=None,methods=None,output_dir="./",file_name="tmp"):
    # model = model_class()
    if not os.path.exists(output_dir):
       os.mkdir(output_dir)
        
    if methods is None or len(methods)==0 :
        with open(os.path.join(output_dir,file_name+".txt"), 'w') as f:
            print(model,file=f)
        return 
            
    if 'ts' in methods:
        summary(model,(100,40))
        
    if 'tviz' in methods:
        y=model(x)
        output=make_dot(y.mean(),params=dict(model.named_parameters()),
                        show_attrs=True,show_saved=True)
        output.format = 'png'
        output.directory = output_dir
        output.render(file_name+"_torchviz",view=False)
        
    if 'tview' in methods:
        model_graph = draw_graph(model, input_size=x.shape, expand_nested=True, save_graph=True, filename=file_name+"_torchview", directory=output_dir)
        model_graph.visual_graph
        
    try:
        if 'onnx' in methods:
            torch.onnx.export(model,x,os.path.join(output_dir,file_name+".onnx"),verbose=True,opset_version=12)
    except Exception as e:
        print("Error when visulized models by onnx.",e)
        
        
    if 'tinfo' in methods:
        with open(os.path.join(output_dir,file_name+"_info.txt"),'w') as f:
            print(info(model,x.shape),file=f)
            

    # model = Pure_Trans_AE_40()
    # x = torch.randn(12,100,40)
    # writer = SummaryWriter()
    
    # model_graph = draw_graph(model,input_size=x.shape,expand_nested=True,save_graph=True,filename="Pure_Trans_AE_40",directory="./")
    
    # model_graph.visual_graph
    
if __name__ == "__main__":
    # Insert the path into sys.path for importing.
    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    import argparse
    from model.cnn1_ae import CNN1_AE
    from model.CNN2 import CNN2
    from model.DeepLOB import DeepLOB_AE
    from model.linear_autoencoder import Linear_AE
    from model.LSTM import LSTM
    # from model.Pure_Trans_AE_1286 import Pure_Trans_AE
    from model.TransLOB import Trans_LOB
    
    
    
    parser = argparse.ArgumentParser(description='Visualizer for NN models')
    parser.add_argument('--model', choices=['deeplob', 'pure_trans_ae', 'trans_lob','cnn1_ae','cnn2_ae','lstm_ae'],
                        required=True,
                            help="Choose a model from the available options: DeepLOB, Pure_Trans_AE, Trans_LOB.")
        
        # 添加--methods参数，接受一个字符串列表
    parser.add_argument('--methods', nargs='+', choices=['ts', 'tviz', 'tview', 'onnx', 'tinfo'], required=True,
                        help="Choose one or more methods from the available options: ts, tviz, tview, onnx, tinfo.")
    
    
    # 解析命令行参数
    args = parser.parse_args()
    
    model_dct={
        'cnn1_ae':CNN1_AE,
        'cnn2_ae':CNN2_AE,
        'deeplob_ae': DeepLOB_AE,
        'linear_ae': Linear_AE,
        'lstm_ae': LSTM_autoencoder,
        'pure_trans_ae': Pure_Trans_AE,
        'trans_lob': Trans_LOB
    }
    
    x=torch.randn(3,100,40)
    try:
        model_name = args.model.lower()
        model = model_dct[args.model]()
    except Exception as e:
        print(f"Not find the model {args.model}, please check your --model args.",e)
        
    Visualize(model,x,methods=args.methods,output_dir="./doc/models_detail/",file_name=args.model)
    
    
    
    
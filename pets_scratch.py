from multiprocessing import Pool, current_process, Queue
import subprocess

#### TO KILL: run: pkill -9 python


# first, define the machines, including ssh and dir navigation for remote ones, and where the main dataset and aug dataset is located 
class Machine0:
    def __init__(self):
        self.name = "machine_0"
        self.aug_dir = "/home/ubuntu/control_augmented_images_pets_512"
        self.data_dir = "torch/pets"
        
    def run(self,command):
        out = subprocess.run([ command + "\n"],shell=True) 
        print(out)



class GPU:
    def __init__(self,machine,gpu_id):
        self.machine=machine
        self.gpu_id = gpu_id


# initialize the queue with the GPU ids
queue = Queue()

machine_0 = Machine0()
for i in [0,1,2,3,4,5,6,7]: #+ [0,1,2,3,4,5,6,7]:
    queue.put(GPU(machine_0,i))
num_processes = 8





def get_command_string(command,gpu,aug_directory,data_dir):
    def get_command(device,aug_directory,model,name,experiment,recipe,shots,variations,ways,repeats,seed,data_directory):
        dataset = experiment.split("-")[0]
        return 'CUDA_VISIBLE_DEVICES='+str(device)+' python3 train.py '+data_directory+' --dataset torch/'+dataset+'  --model='+model+'  --num-classes=257  --log-wandb --experiment "'+experiment+'" --diffaug-dir='+aug_directory+' --seed '+str(seed)+'  --name "'+name+'" --recipe "'+recipe+'"  --repeats '+str(repeats)+' --variations '+str(variations)+' --diffaug-fewshot='+str(shots)+' '+ways+' '
     
    command_string = get_command(gpu,aug_directory,command[0],command[1],command[2],command[3],command[4],command[5],command[6],command[7],command[8],data_dir)
    return command_string


def get_full_dataset_commands_aircraft_scratch():
    commands = []


    # define the dataset. Make sure this matches up with the directories given in the machines
    dataset = "pets"    
    
    # define the sweep to do
    recipes = ["sgd-scratch-fullaug"] #,"sgd-scratch-noaug" ]
    seeds = [11,21]#[10,20,30]    
    models= ["resnet101,vit_base_patch8_224"]#["vgg19","efficientnet_b2"]#["resnet50",vit_base_patch8_224]
    
    for model in models:
        for recipe in recipes:
            for seed in seeds:            
                variation=2 
                shot=0
                way_str=""
                experiment = dataset + "-" + "full"                   
                exp_name = "switch exp seed "+str(seed) + model + " " + recipe
                exp_repeats = 2
                
                base_name = "switch base seed"+str(seed) + model + " " + recipe
                base_repeats = 6

                # if "fullaug" in recipe:
                #     way_str = " --valid-nonorm "
                commands.append([model,exp_name,experiment,recipe,shot,variation,way_str+" --switch",exp_repeats,seed])
                commands.append([model,base_name,experiment,recipe,shot,0,way_str,base_repeats,seed])
    return commands
def foo(command):
    gpu = queue.get()
    try:
        print("running on " + gpu.machine.name + ":"+ str(gpu.gpu_id))
        command_string = get_command_string(command,gpu.gpu_id,gpu.machine.aug_dir,gpu.machine.data_dir)
        gpu.machine.run(command_string)
    finally:
        queue.put(gpu)



pool = Pool(processes=num_processes)

commands = get_full_dataset_commands_aircraft_scratch()


for c in commands:
    print(get_command_string(c,0,machine_0.aug_dir,machine_0.data_dir))
    print(len(commands))
    
for _ in pool.imap_unordered(foo, commands):
    pass
pool.close()
pool.join()




from multiprocessing import Pool, current_process, Queue
import subprocess

#### TO KILL: run: pkill -9 python


# first, define the machines, including ssh and dir navigation for remote ones, and where the main dataset and aug dataset is located 
class Machine0:
    def __init__(self):
        self.name = "machine_0"
        self.aug_dir = "/data/puma_envs/control_augmented_images_caltech256_512fewshot/"
        self.data_dir = "/data/torch/caltech256/caltech256"

    def run(self,command):
        out = subprocess.run([ command + "\n"],shell=True) 
        print(out)

class Machine1:
    def __init__(self):
        self.name = "machine_1"
        self.aug_dir = "/home/judah/control_augmented_images_caltech256_512"
        self.data_dir = "torch/caltech256"
        
    def run(self,command):
        out = subprocess.run(["sshpass","-p","rjk2147","ssh", "judah@rjk2147.cuit.columbia.edu","cd diffusion_augmentation \n", command + "\n"]) 
        print(out)

class GPU:
    def __init__(self,machine,gpu_id):
        self.machine=machine
        self.gpu_id = gpu_id


# initialize the queue with the GPU ids
queue = Queue()

machine_0 = Machine0()
for i in [0,1,2,3,4,5,6,0,1,2,3,4,5,6]:
    queue.put(GPU(machine_0,i))
# machine_1 = Machine1()
# for i in range(3):
#     queue.put(GPU(machine_1,i))
num_processes = 14


machine_1 = Machine1()
# for i in range(3):
#     queue.put(GPU(machine_1,i))
# num_processes = 3

def get_fewshot_commands_caltech_scratch():
    commands = []


    # define the dataset. Make sure this matches up with the directories given in the machines
    dataset = "caltech256"    
    
    # define the sweep to do
    recipe = "sgd-scratch-short-fullaug" 
    seeds = [10,20,30]    
    ways = ['all']#["5","10","all"]
    shots = [1,2,5,10]
    variations = [15]#,5,10,15]
    models= ["resnet50"]
    
    for model in models:
        for seed in seeds:
            for way in ways:
                if way=="all":
                    way_str = ""
                if way=="5":
                    way_str = " --classes 0 1 2 3 4 "
                if way=="10":
                    way_str= " --classes 0 1 2 3 4 5 6 7 8 9 "
                for shot in shots:
                    for variation in variations:            
                        experiment = dataset + "-" + way + "-" + str(shot) + "-" + str(variation) + "-scratch"#"-pretrainsun"

                        target_repeats = 128 
                        if ways=="all":
                            target_repeats = 16  
                            
                        exp_name = "exp seed "+str(seed) + model + " " + recipe
                        exp_repeats = target_repeats//(variation+1)
                        
                        base_name = "base seed"+str(seed) + model + " " + recipe
                        base_repeats = exp_repeats * (variation+1)

                        commands.append([model,exp_name,experiment,recipe,shot,variation,way_str,exp_repeats,seed])
                        commands.append([model,base_name,experiment,recipe,shot,0,way_str,base_repeats,seed])
    return commands


def get_fewshot_commands_caltech_pretrain():
    commands = []


    # define the dataset. Make sure this matches up with the directories given in the machines
    dataset = "caltech256"    
    
    # define the sweep to do
    recipe = "sgd-pretrain-fullaug" 
    seeds = [10,20,30]    
    ways = ['all']#["5","10","all"]
    shots = [1,2,5,10]
    variations = [15]#,5,10,15]
    models= ["resnet50sun"]
    
    for model in models:
        for seed in seeds:
            for way in ways:
                if way=="all":
                    way_str = ""
                if way=="5":
                    way_str = " --classes 0 1 2 3 4 "
                if way=="10":
                    way_str= " --classes 0 1 2 3 4 5 6 7 8 9 "
                for shot in shots:
                    for variation in variations:            
                        experiment = dataset + "-" + way + "-" + str(shot) + "-" + str(variation) + "-pretrainsun"

                        target_repeats = 128
                        if ways=="all":
                            target_repeats = 16                        
                        exp_name = "exp seed "+str(seed) + model + " " + recipe
                        exp_repeats = target_repeats//(variation+1)
                        
                        base_name = "base seed"+str(seed) + model + " " + recipe
                        base_repeats = exp_repeats * (variation+1)

                        commands.append([model,exp_name,experiment,recipe,shot,variation,way_str,exp_repeats,seed])
                        commands.append([model,base_name,experiment,recipe,shot,0,way_str,base_repeats,seed])
    return commands



def get_command_string(command,gpu,aug_directory,data_dir):
    def get_command(device,aug_directory,model,name,experiment,recipe,shots,variations,ways,repeats,seed,data_directory):
        return 'CUDA_VISIBLE_DEVICES='+str(device)+' python3 train.py '+data_directory+' --dataset torch/caltech256  --model='+model+'  --num-classes=257  --log-wandb --experiment "'+experiment+'" --diffaug-dir='+aug_directory+' --seed '+str(seed)+'  --name "'+name+'" --recipe "'+recipe+'"  --repeats '+str(repeats)+' --variations '+str(variations)+' --diffaug-fewshot='+str(shots)+' '+ways+'  '
     
    command_string = get_command(gpu,aug_directory,command[0],command[1],command[2],command[3],command[4],command[5],command[6],command[7],command[8],data_dir)
    return command_string






def foo(command):
    gpu = queue.get()
    try:
        print("running on " + gpu.machine.name + ":"+ str(gpu.gpu_id))
        command_string = get_command_string(command,gpu.gpu_id,gpu.machine.aug_dir,gpu.machine.data_dir)
        gpu.machine.run(command_string)
    finally:
        queue.put(gpu)



pool = Pool(processes=num_processes)

commands = get_fewshot_commands_caltech_scratch() + get_fewshot_commands_caltech_pretrain()
for c in commands:
    print(get_command_string(c,0,machine_1.aug_dir,machine_1.data_dir))
    print(len(commands))
    
for _ in pool.imap_unordered(foo, commands):
    pass
pool.close()
pool.join()




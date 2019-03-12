def print_to_log(s, file ='plot_and_log/log.txt'):
    with open(file,'a+') as f:
        f.write(s)

import logging


class Logger:
    def __init__(self, logdir, rank, filename=None, step=None):
        self.logger = None
        self.step = step
        self.logdir = logdir
        self.rank = rank
        logging.basicConfig(filename=filename, level=logging.INFO, format=f'%(levelname)s:%(message)s')
        if rank == 0:
            logging.info(f"[!] starting logging at directory {logdir}")

    def log(self, string):
        '''
        Write one line of log into screen and file.
            log_file_path: Path of log file.
            string:        String to write in log file.
        '''
        if self.rank == 0:
            with open(self.logdir, 'a+') as f:
                f.write(string + '\n')
                f.flush()
            print(string)

    def info(self, msg):
        if self.rank == 0:
            logging.info(msg)

    def error(self, msg):
        if self.rank == 0:
            logging.error(msg)



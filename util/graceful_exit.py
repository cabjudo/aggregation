import signal
import time

class GracefulExitException(Exception):
    pass

class GracefulExit:
    def __init__(self, duration, trainer):
        self.duration = duration
        self.trainer = trainer

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.alarm_handler)
        signal.signal(signal.SIGTERM, self.term_handler)
        
        self.start_time = time.time()
        signal.alarm(self.duration)

        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        # stop alarm
        signal.alarm(0)
        print('leaving context...')
        self.print_duration_info()
        # to suppress exception
        if exc_type == GracefulExitException:
            print('GracefulExit starting clean_up()')
            self.clean_up()
            print('GracefulExitException supressed by context manager')
            return True

    def train(self, *args, **kwargs):
        self.trainer.train(*args, **kwargs)

    def clean_up(self, *args, **kwargs):
        self.trainer.clean_up(*args, **kwargs)

    def alarm_handler(self, signum, stack):
        '''
        Handles an alarm signal
        '''
        print('ALRM recieved')
        self.print_duration_info()
        raise GracefulExitException

    def term_handler(self, signum, stack):
        '''
        Handles term signal sent by kubernetes (hopefully before kill is sent)
        '''
        # stop alarm
        print('disable alarm...')
        signal.alarm(0)
        print('TERM recieved')
        self.print_duration_info()
        raise GracefulExitException

    def print_duration_info(self):
        self.end_time = time.time()
        self.delta = self.end_time - self.start_time
        print('start time: ', self.start_time, ' end_time: ', self.end_time, ' delta: ', self.delta)

        

if __name__ == '__main__':
    some_amt_of_time = 2
    another_amt_of_time = 5

    class DummyTrainer(object):
        def __init__(self, amt_of_time):
            self.amt_of_time = amt_of_time

        def train(self):
            time.sleep(self.amt_of_time)

        def clean_up(self):
            print('saving...')
            time.sleep(10)
            


    dt = DummyTrainer(another_amt_of_time)

    with GracefulExit(some_amt_of_time, dt) as ge:
        print(ge)
        ge.train()




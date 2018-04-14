import signal
import time

class MyException(Exception):
    pass

class GracefulExit:
    def __init__(self, duration):
        self.duration = duration

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.alarm_handler)
        signal.signal(signal.SIGTERM, self.term_handler)
        
        self.start_time = time.time()
        signal.alarm(self.duration)
    
    def __exit__(self, exc_type, exc_value, traceback):
        # stop alarm
        signal.alarm(0)
        print('leaving context...')
        self.print_duration_info()
        # to suppress exception
        if exc_type == MyException:
            print('MyException supressed by context manager')
            return True

    def alarm_handler(self, signum, stack):
        '''
        Handles an alarm signal
        '''
        print('ALRM recieved')
        self.print_duration_info()
        raise MyException

    def term_handler(self, signum, stack):
        '''
        Handles term signal sent by kubernetes (hopefully before kill is sent)
        '''
        # stop alarm
        print('disable alarm...')
        signal.alarm(0)
        print('TERM recieved')
        self.print_duration_info()
        raise MyException

    def print_duration_info(self):
        self.end_time = time.time()
        self.delta = self.end_time - self.start_time
        print('start time: ', self.start_time, ' end_time: ', self.end_time, ' delta: ', self.delta)

        

if __name__ == '__main__':
    some_amt_of_time = 60*2
    another_amt_of_time = 60*5
    
    with GracefulExit(some_amt_of_time):
        try:
            # sleep for 10 seconds
            time.sleep(another_amt_of_time)
        except MyException:
            print('caught MyException')
            print('saving...')
            time.sleep(10)
        finally:
            print('wrap up...')
            time.sleep(5)




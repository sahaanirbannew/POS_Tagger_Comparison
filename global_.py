from datetime import datetime

const_info =  '[ Info]'
const_error = '[Error]'

#                  w,o,m
config_params ={0:[1,0,1],
                1:[1,0,0],
                2:[1,1,1], #default
                3:[1,1,0],
                4:[0,0,1], 
                5:[0,1,1],
                6:[0,1,0]}


def log_entry(text, msg_type):
  log_path = '/content/log.txt'
  try:
    log_file = open(log_path, 'a+')
  except:
      log_file = open('log.txt', 'a+')
  time_string = get_time_string()
  log_file.write(time_string + ' ' +msg_type + ' :' + text+'\n')
  log_file.close()
  print(text)

def get_time_string():
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
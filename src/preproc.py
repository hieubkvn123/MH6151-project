from sklearn.preprocessing import LabelEncoder

def preproc_education(x):
    if(x == 'unknown') : return 0
    elif(x == 'primary') : return 1
    elif(x == 'secondary') : return 2
    elif(x == 'tertiary') : return 3
    
def preproc_month(x):
    if(x == 'jan') : return 1
    elif(x == 'feb') : return 2
    elif(x == 'mar') : return 3
    elif(x == 'apr') : return 4
    elif(x == 'may') : return 5
    elif(x == 'jun') : return 6
    elif(x == 'jul') : return 7
    elif(x == 'aug') : return 8
    elif(x == 'sep') : return 9
    elif(x == 'oct') : return 10
    elif(x == 'nov') : return 11
    elif(x == 'dec') : return 12
    
def preproc_binary(x):
    if(x == 'no') : return 0
    elif(x == 'yes') : return 1
from datetime import datetime
import schedule 

print(datetime.now())
schedule.every().hour.at(":00")
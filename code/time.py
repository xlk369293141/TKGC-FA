from datetime import datetime
from dateutil import relativedelta

# # get two dates
# d1 = '14/8/2019'
# d2 = '16/3/2022'

# # convert string to date object
# start_date = datetime.strptime(d1, "%d/%m/%Y")
# end_date = datetime.strptime(d2, "%d/%m/%Y")

# # Get the relativedelta between two dates
# delta = relativedelta.relativedelta(end_date, start_date)
# print('Years, Months, Days between two dates is')
# print(delta.years, 'Years,', delta.months, 'months,', delta.days, 'days')
# print(start_date,end_date)




# from datetime import datetime

# datetime_list = [
#     datetime(2009, 10, 12, 10, 10),
#     datetime(2010, 10, 12, 10, 10),
#     datetime(2010, 10, 12, 10, 10),
#     datetime(2011, 10, 12, 10, 10), #future
#     datetime(2012, 10, 12, 10, 10), #future
# ]

# oldest = min(datetime_list)
# youngest = max(datetime_list)
# delta = relativedelta.relativedelta(youngest, oldest)
# print(delta.years, 'Years,', delta.months, 'months,', delta.days, 'days')



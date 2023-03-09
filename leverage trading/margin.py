#margin------------------------------------------------------------------margin

#input is a list of buy_price and sell_price in every day in a period of time. 
def margin(buy_price, sell_price, interest, leverage, cashbalance):
    borrowed = 0
    desired_buy_price = buy_price * leverage
    desired_sell_price = sell_price * leverage
    for i in range(len(desired_buy_price)):
        if cashbalance >= desired_buy_price[i]:
            cashbalance-=desired_buy_price[i]
        else:
            borrowed = borrowed + desired_buy_price[i] - cashbalance
            cashbalance = 0
        borrowed = borrowed * (1+interest)
        cashbalance += desired_sell_price[i]
        if cashbalance >= borrowed:
            cashbalance = cashbalance - borrowed
        else:
            borrowed = borrowed - cashbalance
            cashbalance = 0

    return cashbalance, borrowed


# example
cashbalance, borrowed = margin([1,2,3,4,5,6,7],[2,3,4,4,3,2,1], 0.2, 2, 10)
print('The cash balance is ', cashbalance, '. The money owed is ', borrowed)
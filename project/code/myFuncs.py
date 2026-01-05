def bmi(hgt, wgt:float=70) -> float:
    '''
    This function returns bmi from weight(kg) and height(cm)
    '''
    return wgt / (hgt / 100) ** 2
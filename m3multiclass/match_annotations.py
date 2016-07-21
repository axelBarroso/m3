
def matchAnnotations(annotation):
#Converts from initial annotations to 15 types

    if annotation in ['A13', 'A14', 'A15', 'A1A', 'A1B', 'A1C', 'A1D', 'A23', 'A25', 'A29', 'A30', 'A33', 'A41', 'A51', 'A7A', 'A7B', 'A7C']:
        return '0'

    if annotation in ['B15A', 'B1', 'B19', 'B11']:
        return '1'

    if annotation in [ 'B17']:
        return '2'

    if annotation in ['B21']:
        return '3'

    if annotation in ['B3']:
        return '4'

    if annotation in ['C1']:
        return '5'

    if annotation in ['C3']:
        return '6'

    if annotation in ['C11', 'C21', 'C23', 'C27', 'C29', 'C31LEFT', 'C31RIGHT', 'C35', 'C43']:
        return '7'

    if annotation in ['D1a', 'D10', 'D1b', 'D3b', 'D5', 'D7', 'D9']:
        return '8'

    if annotation in ['E1']:
        return '9'

    if annotation in ['E3']:
        return '10'

    if annotation in ['E7']:
        return '11'

    if annotation in ['E9a', 'E9a_miva', 'E9b', 'E9c', 'E9d', 'E9e']:
        return '12'

    if annotation in ['F12a', 'F12b', 'F45', 'F47', 'F49', 'F50', 'F59', 'F87']:
        return '13'

    return '14' #Is background


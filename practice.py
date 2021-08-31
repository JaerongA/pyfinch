blueShirtSpeeds = [3, 4, 4, 1, 1, 8, 9]
redShirtSpeeds = [9, 8, 2, 2, 3, 5, 6]
fastest =  False

def tandemBicycle(redShirtSpeeds, blueShirtSpeeds, fastest):

    if fastest:
        redShirtSpeeds.sort()
        blueShirtSpeeds.sort()
    else:
        redShirtSpeeds.sort(reverse=True)
        blueShirtSpeeds.sort(reverse=True)

    total_speed = 0

    while redShirtSpeeds:
        total_speed += (max(redShirtSpeeds.pop(0), blueShirtSpeeds.pop()))

    return total_speed

tandemBicycle(redShirtSpeeds, blueShirtSpeeds, fastest)
from difflib import SequenceMatcher


def similar(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


# Initializing strings
test_string1 = 'what is ATOM model'.lower()
test_string2 = 'the atom model is an acronym for align traditional organizations model. the atom model has 4 quadrants. top left: increase revenue (increasing sales to new or existing customers. delighting or disrupting to increase market share and size), top right: protect revenue (improvements and incremental innovation to sustain current market share and revenue figures), bottom-left: reduce costs (costs that you are currently incurring that can be reduced. more efficient, improved margin or contribution), bottom-right: avoid-costs (improvements to sustain current cost base. costs you are not incurring but may do in the future). all decisions in the quadrants need to maintain or increase organizational health. it can be used by leaders, profuct owners, or others to make sustainable decisions and build shared progress.\n\n'.lower()

# using SequenceMatcher.ratio()
# similarity between strings
res = similar(test_string1, test_string2)

# printing the result
print("The similarity between 2 strings is : " + str(res))

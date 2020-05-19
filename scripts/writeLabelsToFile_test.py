import geoid_income_utils

test_dict = {}
test_dict["k1"] = "v1"
test_dict["k2"] = "v2"
test_dict["k3"] = "v3"
test_dict["k4"] = "v4"

geoid_income_utils.writeLabelsToFile("test_output", test_dict)
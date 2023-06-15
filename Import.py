class Import:
    def __init__(self, pID, dir):
        self.pID = pID
        self.dir = dir

    def import_file(self, file_name):
        blocks = list(range(1, 10))
        # NEED TO TEST: pIDs and dir are strings
        for block in blocks:
            block_string = self.dir + self.pID + "r" + str(block)
            file_string = block_string + file_name + ".csv"

    #def trialinfo(self, 'TrialInfo'):
     #   return Import.import_file(self)


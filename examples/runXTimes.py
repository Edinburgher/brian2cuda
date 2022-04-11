from collections.abc import Mapping

from brian2 import BrianObject, Network, NeuronGroup, Synapses


class MultipleNetworkRunner:
    def __init__(self, *objs, X):
        self.objs = objs
        self.X = X
        self.network = Network()

    def checkObjects(self):
        for obj in self.objs:
            if isinstance(obj, BrianObject):
                if obj._network is not None:
                    raise RuntimeError('%s has already been simulated, cannot '
                                       'add it to the network. If you were '
                                       'trying to remove and add an object to '
                                       'temporarily stop it from being run, '
                                       'set its active flag to False instead.'
                                       % obj.name)


    def runXtimes(self):
        self.checkObjects()
        self.addObjects()


    # TODO: this is a bigger problem than anticipated.
    # NeuronGroups can have subgroups
    # Synapses can not just be connected with additional conditions
    #   i/j variant can not be extended, because of the only one for loop restriction
    #   i/j can also not be conbined with condition
    #   condition could be extended with condition, however there is no way to edit connect calls after the fact
    #   therefore we probably need to get the connect call separately
    def addObjects(self):
        for obj in self.objs:
            if isinstance(obj, NeuronGroup):
                obj._N = self.X * obj._N
                self.network.add(obj)
            elif isinstance(obj, Synapses):
                # TODO: see above
                self.network.add(obj)



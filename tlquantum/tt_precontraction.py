import tensorly as tl
from torch import tensordot, transpose
from itertools import chain


# Author: Taylor Lee Patti <taylorpatti@g.harvard.edu>

# License: BSD 3 clause


def layers_contract(layer_list, ncontrl):
    """Contracts sublists of a list of layers vertically (merging cores of multiple layers for a single qubit) up to
    some maximum contraction depth.

    Parameters
    ----------
    layer_list : List of tt-tensor layers
    ncontrl : Maximum number of layers to contract into each composite layer.

    Returns
    -------
    List of layers resulting from vertically contracting the layers of layer_list
    """
    nlayers = len(layer_list)
    contrsets, merge_layers = list(range(nlayers)), []
    contrsets = [contrsets[i:i+ncontrl] for i in range(0, nlayers, ncontrl)]
    for contrset in contrsets:
        merge_layers.append(_vertical_contract([layer_list[i] for i in contrset]))
    merge_layers = list(chain.from_iterable(merge_layers)) #merge list of core lists (layers) into single list
    return merge_layers


def _vertical_contract(layer_list):
    """Contracts list of layers vertically (merging cores of multiple layers for a single qubit).

    Parameters
    ----------
    layer_list : List of tt-tensor layers

    Returns
    -------
    Single layer (list of cores) resulting from vertically contracting the layers of layer_list
    """
    nlayers = len(layer_list)
    nqubits, merge_layers = len(layer_list[0]), []
    for q in range(nqubits):
        cont = layer_list[0][q] #first layer of each qubit to start contraction
        for l in range(1, nlayers):
            dim1, dim2 = cont.shape, layer_list[l][q].shape
            cont = tensordot(layer_list[l][q], cont, dims=([2],[1])) #abdefg --> aedbfg --> aebdfg --> 
            cont = transpose(transpose(transpose(cont, 1, 3), 2, 3), 3, 4)
            cont = tl.reshape(cont, (dim1[0]*dim2[0], dim1[1], dim1[2], dim1[3]*dim2[3]))
        merge_layers.append(cont)
    return merge_layers


def qubits_contract(layer, ncontraq, contrsets=None):
    """Contracts lists (layers) of tt-tensor cores horizontally (merging multiple cores in a single layer) up to
    some maximum number of qubits.

    Parameters
    ----------
    layer : List of tt-tensor cores
    ncontrq : Maximum number of qubits to contract into each composite core.

    Returns
    -------
    Layer (list of cores) resulting from horizontally contracting the cores of layer
    """
    merge_layer, nqubits = [], len(layer)
    if contrsets == None:
        contrsets = _get_contrsets(nqubits, ncontraq)
    if len(layer[0].shape) == 3: #if cores have 3 indices, state tensor
        for contrset in contrsets:
            merge_layer.append(_horizontal_state_contract([layer[i] for i in contrset]))
    else: #if cores do not have 3 indices (here, they would have 4 indices), operator tensor
        for contrset in contrsets:
            merge_layer.append(_horizontal_contract([layer[i] for i in contrset]))
    return merge_layer


def _horizontal_contract(qubits):
    """Contracts list of qubits horizontally (merging multiple cores in a single layer).

    Parameters
    ----------
    qubits : List of tt-tensor cores
    state : tt-tensor, input state to be evolved by unitary

    Returns
    -------
    Single core resulting from horizontally contracting the cores of qubits
    """
    cont = qubits[0]
    for i in range(1, len(qubits)):
        cont = tensordot(cont, qubits[i], dims=([3],[0]))
        cont = transpose(cont, 2, 3)
        cont = tl.reshape(cont, (cont.shape[0], cont.shape[1]*cont.shape[2], cont.shape[3]*cont.shape[4], cont.shape[-1]))
    return cont


def _horizontal_state_contract(qubits):
    """Contracts list of qubits horizontally (merging multiple cores in a single layer).

    Parameters
    ----------
    qubits : List of tt-tensor cores
    state : tt-tensor, input state to be evolved by unitary
    
    Returns
    -------
    Single core resulting from horizontally contracting the cores of qubits
    """
    cont = qubits[0]
    for i in range(1, len(qubits)):
        cont = tensordot(cont, qubits[i], dims=([2],[0]))
        cont = tl.reshape(cont, (cont.shape[0], cont.shape[1]*cont.shape[2], cont.shape[3]))
    return cont


def _get_contrsets(nqubits, ncontraq):
    contrsets = list(range(nqubits))
    return [contrsets[i:i+ncontraq] for i in range(0, nqubits, ncontraq)]

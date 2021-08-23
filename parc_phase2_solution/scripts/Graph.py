#!/usr/bin/env python
import numpy as np


class Vertex:
    def __init__(self, node, coordinates):
        """The initial method of the class

        Args:
            node (str): The identifier of the node
            coordinate (tuple[float, float]): The position coordinates of the node.
        """
        self.id = node
        self.adjacent = {}
        self.coordinates = coordinates

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def __repr__(self):
        return self.__str__()

    def add_neighbor(self, neighbor, how = "lane", weight = 0):
        """Adds a neighbor with weight to the the node

        Args:
            neighbor (str): The neighbor as a string
            how (str, optional): Describes how to move to neighbor. Options are lane and crossing. Defaults to "lane".
            weight (float, optional): The weight of moving to that neighbour. Defaults to 0.
        """
        self.adjacent[neighbor] = (weight, how)

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]


class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.num_vertices = 0

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node, coordinates):
        self.num_vertices = self.num_vertices + 1

        new_vertex = Vertex(node, coordinates)

        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, how):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        frm_node = self.vert_dict[frm]
        to_node = self.vert_dict[to]

        frm_coords = frm_node.coordinates
        to_coords = to_node.coordinates

        distance = np.sqrt(
            (frm_coords[0] - to_coords[0])**2 + (frm_coords[1] - to_coords[1])**2)

        frm_node.add_neighbor(self.vert_dict[to], how, distance)
        to_node.add_neighbor(self.vert_dict[frm], how, distance)

    def get_vertices(self):
        return self.vert_dict.keys()

class BinaryTreeNode:
  def __init__(self, id, value):
    self._id = id
    self._value = value
    self._parent = None
    self._left = None
    self._right = None

  def id(self):
    return self._id

  def value(self):
    return self._value

  def __repr__(self):
    return ''.join(('{id: ', str(self._id), ', value: ', str(self._value), '}'))

class BinarySearchTree:
  def __init__(self):
    self._root = None

  def _insert (self, node, ancestor):
    if node._value < ancestor._value:
      if ancestor._left == None:
        node._parent = ancestor
        ancestor._left = node
      else:
        self._insert(node, ancestor._left)
    else:
      if ancestor._right == None:
        node._parent = ancestor
        ancestor._right = node
      else:
        self._insert(node, ancestor._right)

  def insert(self, node):
    if self._root == None:
      self._root = node
    else:
      self._insert(node, self._root)

  def minimum(self, node):
    if node == None:
      return None
    else:
      if node._left == None:
        return node
      else:
        return self.minimum(node._left)

  def _walk_in_order(self, node, nodes):
    if node._left != None:
      self._walk_in_order(node._left, nodes)
    nodes.append(node)
    if node._right != None:
      self._walk_in_order(node._right, nodes)
    return nodes

  def walk_in_order(self):
    return self._walk_in_order(self._root, [])

  def __repr__(self):
    nodes = self.walk_in_order(self._root, [])
    return str(nodes)


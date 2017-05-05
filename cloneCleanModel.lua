
function cloneCleanModel(node)
  print('333'..torch.type(node))
  print('444'..type(node))

  local clonedNode = {}
  if torch.type(node):find('nn.Sequential') then
    clonedNode = nn.Sequential()    
  end

  if torch.type(node):find('nn.ConcatTable') then
    clonedNode = nn.ConcatTable()    
  end

  if torch.type(node):find('nn.CAddTable') then
    clonedNode = nn.CAddTable()    
  end


  if torch.type(node):find('nn.SpatialReflectionPadding') then
    local pad_l, pad_r, pad_t, pad_b = node.pad_l, node.pad_r, node.pad_t, node.pad_b
    clonedNode = nn.SpatialReflectionPadding(pad_l, pad_r, pad_t, pad_b)
  end

  if torch.type(node):find('nn.InstanceNormalization') then
    local nOutput, eps = node.nOutput, node.eps
    clonedNode = nn.InstanceNormalization(nOutput, eps)
    clonedNode.gradWeight = nil
    clonedNode.gradBias = nil
    clonedNode.weight = node.weight:clone()
    clonedNode.bias = node.bias:clone()
  end

  if torch.type(node):find('nn.ReLU') then
    clonedNode = nn.ReLU()
  end

  if torch.type(node):find('nn.SpatialConvolution') then
    local nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH= node.nInputPlane, node.nOutputPlane, node.kW, node.kH, node.dW, node.dH, node.padW, node.padH
    clonedNode = nn.SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
    clonedNode.gradWeight = nil
    clonedNode.gradBias = nil
    clonedNode.weight = node.weight:clone()
    clonedNode.bias = node.bias:clone()

  end

  if torch.type(node):find('nn.ShaveImage') then
    local size = node.size
    clonedNode = nn.ShaveImage(size)    
  end

  if torch.type(node):find('nn.SpatialUpSamplingNearest') then
    local scale_factor = node.scale_factor
    clonedNode = nn.SpatialUpSamplingNearest(scale_factor)
  end

  if torch.type(node):find('nn.Sigmoid') then
    clonedNode = nn.Sigmoid()
  end

  if torch.type(node):find('nn.MulConstant') then
    local constant_scalar, inplace = node.constant_scalar, node.inplace
    clonedNode = nn.MulConstant(constant_scalar, inplace)
  end

  if torch.type(node):find('nn.TotalVariation') then
    local strength = node.strength
    clonedNode = nn.TotalVariation(strength)
  end


  -- Recurse on nodes with 'modules'
  if (node.modules ~= nil) then
    if (type(node.modules) == 'table') then
      for i = 1, #node.modules do
        local child = node.modules[i]
        print('111'..tostring(child))
        print('222'..tostring(clonedNode))
        local clonedChild = cloneCleanModel(child)
        
        clonedNode:add(clonedChild)
      end
    end
  end

  return clonedNode

  -- collectgarbage()
end

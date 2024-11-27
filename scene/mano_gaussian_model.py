# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#

from pathlib import Path
import numpy as np
import torch
from manopth.manolayer import ManoLayer

from .gaussian_model import GaussianModel
from utils.graphics_utils import compute_face_orientation, compute_E_inverse, compute_E_inverse_GA
# from pytorch3d.transforms import matrix_to_quaternion
from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz, rotmat_to_rotvec
# from roma import quat_product, quat_xyzw_to_wxyz, quat_wxyz_to_xyzw, rotvec_to_rotmat, rotmat_to_rotvec
def polar_decomp(m):   # express polar decomposition in terms of singular-value decomposition
    U, S, Vh = torch.linalg.svd(m)
    u = U @ Vh
    p = Vh.T.conj() @ S.diag().to(dtype = m.dtype) @ Vh
    return  u, p

class MANOGaussianModel(GaussianModel):
    def __init__(self, sh_degree : int, sg_degree: int, not_finetune_MANO_params=False, start_frame_idx=0, step_size=10,\
                train_normal=False, tight_pruning=False, train_kinematic=False, DTF=False,
                invT_Jacobian = False,  densification_type='arithmetic_mean', detach_eyeball_geometry = False, \
                train_kinematic_dist = False,detach_boundary = False):
        super().__init__(sh_degree, sg_degree)

        self.mano_model = ManoLayer(mano_root='mano_model', use_pca=True, flat_hand_mean=True, ncomps = 30).cuda()

        self.verts_templete = self.mano_model.th_v_template

        # binding is initialized once the mesh topology is known
        if self.binding is None:
            self.binding = torch.arange(len(self.mano_model.th_faces)).cuda()
            self.binding_counter = torch.ones(len(self.mano_model.th_faces), dtype=torch.int32).cuda()

        self.face_adjacency = None
        self.L_points = None

        self.mano_param = None
        self.mano_param_orig = None
        self.not_finetune_MANO_params = not_finetune_MANO_params
        self.start_frame_idx = start_frame_idx
        self.step_size = step_size
        
        self.train_normal = train_normal
        self.tight_pruning = tight_pruning
        self.train_kinematic = train_kinematic
        self.train_kinematic_dist = train_kinematic_dist
        self.DTF = DTF
        self.invT_Jacobian = invT_Jacobian
        
        self.densification_type = densification_type
        self.detach_eyeball_geometry = detach_eyeball_geometry
        self.detach_boundary = detach_boundary

    def load_meshes(self, train_meshes, test_meshes, tgt_train_meshes, tgt_test_meshes):
        if self.mano_param is None:
            meshes = {**train_meshes, **test_meshes}
            tgt_meshes = {**tgt_train_meshes, **tgt_test_meshes}
            pose_meshes = meshes if len(tgt_meshes) == 0 else tgt_meshes
             
            self.num_timesteps = len(pose_meshes)  # required by viewers
            num_verts = self.mano_model.th_v_template.shape[0]

            T = self.num_timesteps

            self.mano_param = {
                'shape': torch.zeros([T, meshes[0].shape.shape[1]]),
                'pose': torch.zeros([T, meshes[0].pose.shape[0]]),
                'trans': torch.zeros([T, 3]),
                'scale': torch.zeros([T, 1])
            }

            for i, mesh in pose_meshes.items():
                self.mano_param['shape'][i] = torch.from_numpy(mesh.shape)
                self.mano_param['pose'][i] = torch.from_numpy(mesh.pose)
                self.mano_param['trans'][i] = torch.from_numpy(mesh.trans)
                self.mano_param['scale'][i] = torch.from_numpy(mesh.scale)
            
            for k, v in self.mano_param.items():
                self.mano_param[k] = v.float().cuda()

            self.mano_param_orig = {k: v.clone() for k, v in self.mano_param.items()}
        else:
            # NOTE: not sure when this happens
            import ipdb; ipdb.set_trace()
            pass
    
    def select_mesh_by_timestep(self, timestep, original=False):
        def save_verts_as_ply(verts, name):
            import trimesh
            verts = verts.squeeze(0).detach().cpu().numpy()
            faces = self.mano_model.th_faces.cpu().numpy()
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh.export(f"vis/mano_mesh/mesh_{name}.ply")
        def average_edge_length(vertices):
            edges = set()
            vertices = vertices.squeeze(0).detach().cpu().numpy()
            faces = self.mano_model.th_faces.cpu().numpy()

            # Loop through each face (triangle)
            for face in faces:
                # Get the three edges of the triangle
                for i in range(3):
                    v1 = face[i]
                    v2 = face[(i + 1) % 3]  # Wraps around to form a triangle
                    # Store edges as tuples (smaller index first for uniqueness)
                    edge = tuple(sorted([v1, v2]))
                    edges.add(edge)

            # Calculate the length of each unique edge
            total_length = 0.0
            for edge in edges:
                p1, p2 = vertices[edge[0]], vertices[edge[1]]
                length = np.linalg.norm(p1 - p2)
                total_length += length

            # Calculate the average length
            average_length = total_length / len(edges)

            return average_length
        
        self.timestep = timestep
        mano_param = self.mano_param_orig if original and self.mano_param_orig != None else self.mano_param
        # print(mano_param['shape'][0])
        verts, joints, bone_T, pose_delta, shape_delta, verts_cano = self.mano_model(
            mano_param['pose'][timestep].unsqueeze(0),
            mano_param['shape'][timestep].unsqueeze(0),
        )

        # verts: posed+shaped hand mesh in timestep
        # verts_cano: posed+shaped hand mesh in canical space
        # sefl.verts_templete: no pose no shape, templete mesh in canonical

        verts = (verts/1000) * mano_param['scale'][timestep] + mano_param['trans'][timestep]   
        joints = (joints/1000) * mano_param['scale'][timestep] + mano_param['trans'][timestep]
        verts_cano = (verts_cano/1000) * mano_param['scale'][timestep] + mano_param['trans'][timestep]

        self.update_mesh_properties(verts, verts_cano)
    
    def apply_mano_param(self, mano_param):
        verts, joints, bone_T, pose_delta, shape_delta, verts_cano = self.mano_model(
            mano_param['pose'],
            mano_param['shape'][0],
        )
        verts = (verts/1000) * mano_param['scale'][0] + mano_param['trans'][0]   
        joints = (joints/1000) * mano_param['scale'][0] + mano_param['trans'][0]
        verts_cano = (verts_cano/1000) * mano_param['scale'][0] + mano_param['trans'][0]

        self.update_mesh_properties(verts, verts_cano)

    def update_mesh_properties(self, verts, verts_cano):
        faces = self.mano_model.th_faces
        triangles = verts[:, faces]
        self.face_center = triangles.mean(dim=-2).squeeze(0)

        # orientation and scale
        # if self.scaling_aniso:
        #     return_dim = 2
        # else:
        return_dim = 1
        # breakpoint()
        if self.DTF:
            return_type = 'DTF'
        else:
            return_type = 'GA'

        if return_type == 'GA':
            self.face_orien_mat, self.face_scaling = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), \
                            return_scale=True, scale_dim =return_dim, return_type = return_type)
            # self.face_orien_quat = matrix_to_quaternion(self.face_orien_mat)  # pytorch3d (WXYZ)
            self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat))  # roma
                
        else:
            self.face_orien_mat, self.face_scaling = compute_face_orientation(verts_cano.squeeze(0), faces.squeeze(0), \
                            return_scale=True,scale_dim =return_dim, return_type = 'GA')
            
            self.face_orien_quat = quat_xyzw_to_wxyz(rotmat_to_unitquat(self.face_orien_mat)) 
           
            self.E_inverse = compute_E_inverse(verts_cano.squeeze(0), faces.squeeze(0))


            def polar_decomp(m):   # express polar decomposition in terms of singular-value decomposition
            #! this convention 
                #! https://blog.naver.com/PostView.naver?blogId=richscskia&logNo=222179474476
                #! https://discuss.pytorch.org/t/polar-decomposition-of-matrices-in-pytorch/188458/2
                # breakpoint()
                U, S, Vh = torch.linalg.svd(m)
                U_new = torch.bmm(U, Vh) #! Unitary
                P = torch.bmm(torch.bmm(Vh.permute(0,2,1).conj(), torch.diag_embed(S).to(dtype = m.dtype)), Vh) #! PSD
                # P = torch.bmm(torch.bmm(Vh.permute(0,2,1).conj(), torch.diag_embed(S)), Vh)
                return U_new, P
            self.face_trans_mat = compute_face_orientation(verts.squeeze(0), faces.squeeze(0), \
                            return_scale=True,scale_dim =return_dim, return_type = return_type, E_inverse = self.E_inverse)
            self.face_R_mat, self.face_U_mat = polar_decomp(self.face_trans_mat)

            self.blended_Jacobian = None
            self.blended_R = None
            self.blended_U = None
            # print("FLUSHED, Blended Jacobian,R and U")
            # self.R_rotvec = rotmat_to_rotvec(self.R_mat).detach()           
        
           
           
        #! Q = V^tilda *V^-1 = V^tilda * I = V^tilda == Jacobian
        # for mesh rendering
        self.verts = verts
        self.faces = faces
        #* 
        #?
        # for mesh regularization
        self.verts_cano = verts_cano
        #! 0602 compute adjacent triangle
     
        if self.face_adjacency is None:
            print('Calculating Face Adjacency!')
            num_faces = faces.size(0)
            edge_to_faces = {}
            
            def add_edge(face_idx, v1, v2):
                edge = tuple(sorted([v1.item(), v2.item()]))
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
            
            # Add all edges to the dictionary
            for i, face in enumerate(faces):
                add_edge(i, face[0], face[1])
                add_edge(i, face[1], face[2])
                add_edge(i, face[2], face[0])
            
            # Initialize adjacency tensor
            face_adjacency = torch.full((num_faces, 3), -1, dtype=torch.long)
            
            # Populate adjacency tensor
            for edge, face_list in edge_to_faces.items():
                if len(face_list) > 1:
                    for face in face_list:
                        adj_faces = set(face_list) - {face}
                        for adj_face in adj_faces:
                            if -1 in face_adjacency[face]:
                                idx = (face_adjacency[face] == -1).nonzero(as_tuple=True)[0][0]
                                face_adjacency[face, idx] = adj_face
         
            face_adjacency_identity = torch.arange(face_adjacency.shape[0])
       
            face_adjacency[face_adjacency == -1] = face_adjacency_identity.view(-1, 1).repeat(1, 3)[face_adjacency == -1]
            
            self.face_adjacency = torch.cat([face_adjacency_identity[...,None],face_adjacency],-1)

    def compute_dynamic_offset_loss(self):
        print("Not implemented dynamic offset loss")
        exit()
        # loss_dynamic = (self.flame_param['dynamic_offset'][[self.timestep]] - self.flame_param_orig['dynamic_offset'][[self.timestep]]).norm(dim=-1)
        loss_dynamic = self.flame_param['dynamic_offset'][[self.timestep]].norm(dim=-1)
        return loss_dynamic.mean()
    
    def compute_laplacian_loss(self):
        print("Not implemented laplacian loss")
        exit()
        # offset = self.flame_param['static_offset'] + self.flame_param['dynamic_offset'][[self.timestep]]
        offset = self.flame_param['dynamic_offset'][[self.timestep]]
        verts_wo_offset = (self.verts_cano - offset).detach()
        verts_w_offset = verts_wo_offset + offset

        L = self.flame_model.laplacian_matrix[None, ...].detach()  # (1, V, V)
        lap_wo = L.bmm(verts_wo_offset).detach()
        lap_w = L.bmm(verts_w_offset)
        diff = (lap_wo - lap_w) ** 2
        diff = diff.sum(dim=-1, keepdim=True)
        return diff.mean()
    
    def training_setup(self, training_args):#! 2
        super().training_setup(training_args)

        if self.not_finetune_MANO_params:
            print("Error: MANO parameters are not finetuned!")
            return

        # # # shape
        # self.mano_param['shape'].requires_grad = True
        # param_shape = {'params': [self.mano_param['shape']], 'lr': training_args.mano_shape_lr, "name": "shape"}
        # self.optimizer.add_param_group(param_shape)

        # # pose
        # self.mano_param['pose'].requires_grad = True
        # param_pose = {'params': [self.mano_param['pose']], 'lr': training_args.mano_pose_lr, "name": "pose"}
        # self.optimizer.add_param_group(param_pose)

        # self.mano_param['trans'].requires_grad = True
        # param_trans = {'params': [self.mano_param['trans']], 'lr': training_args.mano_trans_lr, "name": "trans"}
        # self.optimizer.add_param_group(param_trans)

        # self.mano_param['scale'].requires_grad = True
        # param_scale = {'params': [self.mano_param['scale']], 'lr': training_args.mano_scale_lr, "name": "scale"}
        # self.optimizer.add_param_group(param_scale)

    def save_ply(self, path):
        super().save_ply(path)

        npz_path = Path(path).parent / "mano_param.npz"
        mano_param = {k: v.cpu().numpy() for k, v in self.mano_param.items()}
        
        np.savez(str(npz_path), **mano_param)

    def load_ply(self, path, **kwargs):
        super().load_ply(path)

        if not kwargs['has_target']:
            # When there is no target motion specified, use the finetuned FLAME parameters.
            # This operation overwrites the FLAME parameters loaded from the dataset.
            npz_path = Path(path).parent / "mano_param.npz"
            mano_param = np.load(str(npz_path))
            mano_param = {k: torch.from_numpy(v).cuda() for k, v in mano_param.items()}

            self.mano_param = mano_param
            self.num_timesteps = self.mano_param['shape'].shape[0]  # required by viewers
        
        if 'motion_path' in kwargs and kwargs['motion_path'] is not None:
            # When there is a motion sequence specified, load only dynamic parameters.
            motion_path = Path(kwargs['motion_path'])
            mano_param = np.load(str(motion_path))
            mano_param = {k: torch.from_numpy(v).cuda() for k, v in mano_param.items() if v.dtype == np.float32}

            self.mano_param['shape'] = mano_param['shape']
            self.mano_param['pose'] = mano_param['pose']
            self.mano_param['trans'] = mano_param['trans']
            self.mano_param['scale'] = mano_param['scale']
            self.num_timesteps = self.mano_param['shape'].shape[0]  # required by viewers
        
        if 'disable_fid' in kwargs and len(kwargs['disable_fid']) > 0:
            mask = (self.binding[:, None] != kwargs['disable_fid'][None, :]).all(-1)

            self.binding = self.binding[mask]
            self._xyz = self._xyz[mask]
            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]
            self._opacity = self._opacity[mask]



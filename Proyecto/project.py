import customtkinter
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import math
import numpy as np

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"


def Eaa2rotM(angle, axis):
    '''
    Returns the rotation matrix R able to rotate vectors an angle 'angle' (in rads) about the axis 'axis'
    Axis = X Y Z
    '''

    
    axis_norm = np.linalg.norm(axis)
    

    if axis_norm > 1:
        axis = axis / axis_norm

    if axis.ndim == 1:
        axis = axis.reshape((-1, 1))
    

    R = np.eye(3) * np.cos(np.radians(angle)) + (1 - np.cos(np.radians(angle))) * np.outer(axis, axis) + np.sin(np.radians(angle)) * np.array([[0, -axis[2, 0], axis[1, 0]], [axis[2, 0], 0, -axis[0, 0]], [-axis[1, 0], axis[0, 0], 0]])

    return R


def eAngles2rotM(yaw,pitch,roll): #psi ψ, theta θ , phi φ
    '''
    Given a set of Euler angles returns the rotation matrix R
    '''
    ψ = yaw * np.pi/180;
    θ = pitch* np.pi/180;
    φ = roll* np.pi/180;

    R = np.array([[ np.cos(θ) * np.cos(ψ), np.cos(ψ)*np.sin(θ)*np.sin(φ) - np.cos(φ)*np.sin(ψ), np.cos(ψ)*np.cos(φ)*np.sin(θ) + np.sin(ψ)*np.sin(φ)],
                  [ np.cos(θ) * np.sin(ψ), np.sin(ψ)*np.sin(θ)*np.sin(φ) + np.cos(φ)*np.cos(ψ), np.sin(ψ)*np.sin(θ)*np.cos(φ) - np.cos(ψ)*np.sin(φ)],
                  [ -np.sin(θ), np.cos(θ)*np.sin(φ), np.cos(θ)*np.cos(φ)]
                  ])



    return R


def axis_angle_to_quaternion(angle, axis):
    # Normalize the axis vector
    axis = axis / np.linalg.norm(axis)
    
    # Calculate half-angle
    angle /= 2.0
    
    # Calculate quaternion components
    w = np.cos(angle)
    xyz = axis * np.sin(angle)
    
    return np.array([w, *xyz])


def quaternion_rotation_matrix(Q):


    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]


    r00 = q0**2 + q1**2 - q2**2 - q3**2
    r01 = 2 * (q1*q2) - 2*(q0*q3)
    r02 = 2 * (q1*q3) + 2*(q0*q2)


    r10 = 2 * (q1*q2) + 2*(q0*q3)
    r11 = q0**2 - q1**2 + q2**2 - q3**2
    r12 = 2 * (q2*q3) - 2*(q0*q1)


    r20 = 2 * (q1*q3) - 2*(q0*q2)
    r21 = 2 * (q2*q3) + 2*(q0*q1)
    r22 = q0**2 - q1**2 - q2**2 + q3**2


    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def to_quaternion(roll, pitch, yaw):
    # Abbreviations for the various angular functions
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return qw, qx, qy, qz

class Arcball(customtkinter.CTk):

    def __init__(self):
        super().__init__()

        # Orientation vars. Initialized to represent 0 rotation
        self.quat = np.array([[1],[0],[0],[0]])
        self.rotM = np.eye(3)
        self.AA = {"axis": np.array([[0],[0],[0]]), "angle":0.0}
        self.rotv = np.array([[0],[0],[0]])
        self.euler = np.array([[0],[0],[0]])

        # configure window
        self.title("Holroyd's arcball")
        self.geometry(f"{1100}x{580}")
        self.resizable(False, False)

        self.grid_columnconfigure((0,1), weight=0   )
        self.grid_rowconfigure((0,1), weight=1)
        self.grid_rowconfigure(2, weight=0)

        # Cube plot
        self.init_cube()

        self.canvas = FigureCanvasTkAgg(self.fig, self)  # A tk.DrawingArea.
        self.bm = BlitManager(self.canvas,[self.facesObj])
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.pressed = False #Bool to bypass the information that mouse is clicked
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_release_event', self.onrelease)
        
        # Reset button
        self.resetbutton = customtkinter.CTkButton(self, text="Reset", command=self.resetbutton_pressed)
        self.resetbutton.grid(row=3, column=0, padx=(0, 0), pady=(5, 20), sticky="ns")
        
        # Selectable atti
        self.tabview = customtkinter.CTkTabview(self, width=150, height=150)
        self.tabview.grid(row=0, column=1, padx=(0, 20), pady=(20, 0), sticky="nsew")
        self.tabview.add("Axis angle")
        self.tabview.add("Rotation vector")
        self.tabview.add("Euler angles")
        self.tabview.add("Quaternion")

        # Selectable atti: AA
        self.tabview.tab("Axis angle").grid_columnconfigure(0, weight=0)  # configure grid of individual tabs
        self.tabview.tab("Axis angle").grid_columnconfigure(1, weight=0)  # configure grid of individual tabs

        self.label_AA_axis= customtkinter.CTkLabel(self.tabview.tab("Axis angle"), text="Axis:")
        self.label_AA_axis.grid(row=0, column=0, rowspan=3, padx=(80,0), pady=(45,0), sticky="e")

        self.entry_AA_ax1 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax1.insert(0,"1.0")
        self.entry_AA_ax1.grid(row=0, column=1, padx=(5, 0), pady=(50, 0), sticky="ew")

        self.entry_AA_ax2 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax2.insert(0,"0.0")
        self.entry_AA_ax2.grid(row=1, column=1, padx=(5, 0), pady=(5, 0), sticky="ew")

        self.entry_AA_ax3 = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_ax3.insert(0,"0.0")
        self.entry_AA_ax3.grid(row=2, column=1, padx=(5, 0), pady=(5, 10), sticky="ew")

        self.label_AA_angle = customtkinter.CTkLabel(self.tabview.tab("Axis angle"), text="Angle:")
        self.label_AA_angle.grid(row=3, column=0, padx=(120,0), pady=(10, 20),sticky="w")
        self.entry_AA_angle = customtkinter.CTkEntry(self.tabview.tab("Axis angle"))
        self.entry_AA_angle.insert(0,"0.0")
        self.entry_AA_angle.grid(row=3, column=1, padx=(5, 0), pady=(0, 10), sticky="ew")

        self.button_AA = customtkinter.CTkButton(self.tabview.tab("Axis angle"), text="Apply", command=self.apply_AA, width=180)
        self.button_AA.grid(row=5, column=0, columnspan=2, padx=(0, 0), pady=(5, 0), sticky="e")

        # Selectable atti: rotV
        self.tabview.tab("Rotation vector").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Rotation vector").grid_columnconfigure(1, weight=0)
        
        self.label_rotV= customtkinter.CTkLabel(self.tabview.tab("Rotation vector"), text="rot. Vector:")
        self.label_rotV.grid(row=0, column=0, rowspan=3, padx=(2,0), pady=(45,0), sticky="e")

        self.entry_rotV_1 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_1.insert(0,"0.0")
        self.entry_rotV_1.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_rotV_2 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_2.insert(0,"0.0")
        self.entry_rotV_2.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_rotV_3 = customtkinter.CTkEntry(self.tabview.tab("Rotation vector"))
        self.entry_rotV_3.insert(0,"0.0")
        self.entry_rotV_3.grid(row=2, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_rotV = customtkinter.CTkButton(self.tabview.tab("Rotation vector"), text="Apply", command=self.apply_rotV, width=180)
        self.button_rotV.grid(row=5, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Selectable atti: Euler angles
        self.tabview.tab("Euler angles").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Euler angles").grid_columnconfigure(1, weight=0)
        
        self.label_EA_roll= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="roll:")
        self.label_EA_roll.grid(row=0, column=0, padx=(2,0), pady=(50,0), sticky="e")

        self.label_EA_pitch= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="pitch:")
        self.label_EA_pitch.grid(row=1, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_EA_yaw= customtkinter.CTkLabel(self.tabview.tab("Euler angles"), text="yaw:")
        self.label_EA_yaw.grid(row=2, column=0, rowspan=3, padx=(2,0), pady=(5,10), sticky="e")

        self.entry_EA_roll = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_roll.insert(0,"0.0")
        self.entry_EA_roll.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_EA_pitch = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_pitch.insert(0,"0.0")
        self.entry_EA_pitch.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_EA_yaw = customtkinter.CTkEntry(self.tabview.tab("Euler angles"))
        self.entry_EA_yaw.insert(0,"0.0")
        self.entry_EA_yaw.grid(row=2, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_EA = customtkinter.CTkButton(self.tabview.tab("Euler angles"), text="Apply", command=self.apply_EA, width=180)
        self.button_EA.grid(row=5, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Selectable atti: Quaternion
        self.tabview.tab("Quaternion").grid_columnconfigure(0, weight=1)
        self.tabview.tab("Quaternion").grid_columnconfigure(1, weight=0)
        
        self.label_quat_0= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q0:")
        self.label_quat_0.grid(row=0, column=0, padx=(2,0), pady=(50,0), sticky="e")

        self.label_quat_1= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q1:")
        self.label_quat_1.grid(row=1, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_quat_2= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q2:")
        self.label_quat_2.grid(row=2, column=0, padx=(2,0), pady=(5,0), sticky="e")

        self.label_quat_3= customtkinter.CTkLabel(self.tabview.tab("Quaternion"), text="q3:")
        self.label_quat_3.grid(row=3, column=0, padx=(2,0), pady=(5,10), sticky="e")

        self.entry_quat_0 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_0.insert(0,"1.0")
        self.entry_quat_0.grid(row=0, column=1, padx=(5, 60), pady=(50, 0), sticky="ew")

        self.entry_quat_1 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_1.insert(0,"0.0")
        self.entry_quat_1.grid(row=1, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_quat_2 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_2.insert(0,"0.0")
        self.entry_quat_2.grid(row=2, column=1, padx=(5, 60), pady=(5, 0), sticky="ew")

        self.entry_quat_3 = customtkinter.CTkEntry(self.tabview.tab("Quaternion"))
        self.entry_quat_3.insert(0,"0.0")
        self.entry_quat_3.grid(row=3, column=1, padx=(5, 60), pady=(5, 10), sticky="ew")

        self.button_quat = customtkinter.CTkButton(self.tabview.tab("Quaternion"), text="Apply", command=self.apply_quat, width=180)
        self.button_quat.grid(row=4, column=0, columnspan=2, padx=(0, 60), pady=(5, 0), sticky="e")

        # Rotation matrix info
        self.RotMFrame = customtkinter.CTkFrame(self, width=150)
        self.RotMFrame.grid(row=1, column=1, rowspan=3, padx=(0, 20), pady=(20, 20), sticky="nsew")

        self.RotMFrame.grid_columnconfigure((0,1,2,3,4), weight=1)

        self.label_RotM= customtkinter.CTkLabel(self.RotMFrame, text="RotM = ")
        self.label_RotM.grid(row=0, column=0, rowspan=3, padx=(2,0), pady=(20,0), sticky="e")

        self.entry_RotM_11= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_11.insert(0,"1.0")
        self.entry_RotM_11.configure(state="disabled")
        self.entry_RotM_11.grid(row=0, column=1, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_12= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_12.insert(0,"0.0")
        self.entry_RotM_12.configure(state="disabled")
        self.entry_RotM_12.grid(row=0, column=2, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_13= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_13.insert(0,"0.0")
        self.entry_RotM_13.configure(state="disabled")
        self.entry_RotM_13.grid(row=0, column=3, padx=(2,0), pady=(20,0), sticky="ew")

        self.entry_RotM_21= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_21.insert(0,"0.0")
        self.entry_RotM_21.configure(state="disabled")
        self.entry_RotM_21.grid(row=1, column=1, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_22= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_22.insert(0,"1.0")
        self.entry_RotM_22.configure(state="disabled")
        self.entry_RotM_22.grid(row=1, column=2, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_23= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_23.insert(0,"0.0")
        self.entry_RotM_23.configure(state="disabled")
        self.entry_RotM_23.grid(row=1, column=3, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_31= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_31.insert(0,"0.0")
        self.entry_RotM_31.configure(state="disabled")
        self.entry_RotM_31.grid(row=2, column=1, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_32= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_32.insert(0,"0.0")
        self.entry_RotM_32.configure(state="disabled")
        self.entry_RotM_32.grid(row=2, column=2, padx=(2,0), pady=(2,0), sticky="ew")

        self.entry_RotM_33= customtkinter.CTkEntry(self.RotMFrame, width=50, border_width=0)
        self.entry_RotM_33.insert(0,"1.0")
        self.entry_RotM_33.configure(state="disabled")
        self.entry_RotM_33.grid(row=2, column=3, padx=(2,0), pady=(2,0), sticky="ew")

        self.lastM = np.array([0,0,0])
        self.rotM = np.eye(3)
    


    def setRotMatrix(self, rotM):
        """
        Funcion que hace cambiar la rotM de abajo
        """

        rotM = rotM.copy();
       
        self.entry_RotM_11.configure(state="normal")
        self.entry_RotM_12.configure(state="normal")
        self.entry_RotM_13.configure(state="normal")
        self.entry_RotM_21.configure(state="normal")
        self.entry_RotM_22.configure(state="normal")
        self.entry_RotM_23.configure(state="normal")
        self.entry_RotM_31.configure(state="normal")
        self.entry_RotM_32.configure(state="normal")
        self.entry_RotM_33.configure(state="normal")

        self.entry_RotM_11.delete(0, 99)
        self.entry_RotM_12.delete(0, 99)
        self.entry_RotM_13.delete(0, 99)
        self.entry_RotM_21.delete(0, 99)
        self.entry_RotM_22.delete(0, 99)
        self.entry_RotM_23.delete(0, 99)
        self.entry_RotM_31.delete(0, 99)
        self.entry_RotM_32.delete(0, 99)
        self.entry_RotM_33.delete(0, 99)

        
        if(abs(rotM[0][0]) < 1.0e-16):
            rotM[0][0] = 0
        if(abs(rotM[0][1]) < 1.0e-16):
            rotM[0][1] = 0
        if(abs(rotM[0][2]) < 1.0e-16):
            rotM[0][2] = 0
        if(abs(rotM[1][0]) < 1.0e-16):
            rotM[1][0] = 0
        if(abs(rotM[1][1]) < 1.0e-16):
            rotM[1][1] = 0
        if(abs(rotM[1][2]) < 1.0e-16):
            rotM[1][2] = 0
        if(abs(rotM[2][0]) < 1.0e-16):
            rotM[2][0] = 0
        if(abs(rotM[2][1]) < 1.0e-16):
            rotM[2][1] = 0
        if(abs(rotM[2][2]) < 1.0e-16):
            rotM[2][2] = 0
        

        

        self.entry_RotM_11.insert(0,rotM[0][0])
        self.entry_RotM_12.insert(0,rotM[0][1])
        self.entry_RotM_13.insert(0,rotM[0][2])
        self.entry_RotM_21.insert(0,rotM[1][0])
        self.entry_RotM_22.insert(0,rotM[1][1])
        self.entry_RotM_23.insert(0,rotM[1][2])
        self.entry_RotM_31.insert(0,rotM[2][0])
        self.entry_RotM_32.insert(0,rotM[2][1])
        self.entry_RotM_33.insert(0,rotM[2][2])

        self.entry_RotM_11.configure(state="disabled")
        self.entry_RotM_12.configure(state="disabled")
        self.entry_RotM_13.configure(state="disabled")
        self.entry_RotM_21.configure(state="disabled")
        self.entry_RotM_22.configure(state="disabled")
        self.entry_RotM_23.configure(state="disabled")
        self.entry_RotM_31.configure(state="disabled")
        self.entry_RotM_32.configure(state="disabled")
        self.entry_RotM_33.configure(state="disabled")

    def rotMToAngleAxis(self, rotM):

        
        R = rotM

        angle = np.arccos((np.trace(R) - 1) / 2)
        if(angle != 0):
            axis = (R - R.T) / (2 * np.sin(angle))
            axis_org = np.array([axis[2, 1], axis[0, 2], axis[1, 0]])
        else:
            axis_org = np.array([1,0,0])

        angle = np.rad2deg(angle)

        self.entry_AA_angle.delete(0,99)
        self.entry_AA_angle.insert(0, angle)

        self.entry_AA_ax1.delete(0,99);
        self.entry_AA_ax2.delete(0,99);
        self.entry_AA_ax3.delete(0,99);
        self.entry_AA_ax1.insert(0,axis_org[0]);
        self.entry_AA_ax2.insert(0,axis_org[1]);
        self.entry_AA_ax3.insert(0,axis_org[2]);


        
    
    def rotMToRotationVector(self, rotM):
        R = rotM

        angle = np.arccos((np.trace(R) - 1) / 2)

        


        if(angle != 0):
            axis = (R - R.T) / (2 * np.sin(angle))
            axis_org = np.array([axis[2, 1], axis[0, 2], axis[1, 0]])
        else:
            axis_org = np.array([1,0,0])

        rotV = axis_org * np.rad2deg(angle);


       
        self.entry_rotV_1.delete(0,99);
        self.entry_rotV_2.delete(0,99);
        self.entry_rotV_3.delete(0,99);
        self.entry_rotV_1.insert(0,rotV[0]);
        self.entry_rotV_2.insert(0,rotV[1]);
        self.entry_rotV_3.insert(0,rotV[2]);
        
    
    def rotMToEuler(self, rotM):
        R = rotM

        pitch = np.arcsin(-R[2][0])
        yaw = np.arctan2(R[1][0]/np.cos(pitch), R[0][0]/np.cos(pitch))
        roll = np.arctan2(R[2][1]/np.cos(pitch), R[2][2]/np.cos(pitch))

        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)
        roll = np.rad2deg(roll)

        self.entry_EA_pitch.delete(0,99);
        self.entry_EA_yaw.delete(0,99);
        self.entry_EA_roll.delete(0,99);
        self.entry_EA_pitch.insert(0,pitch);
        self.entry_EA_yaw.insert(0,yaw);
        self.entry_EA_roll.insert(0,roll);

    def rotMToQuat(self, rotM):

        R = rotM

        angle = np.arccos((np.trace(R) - 1) / 2)
        if(angle != 0):
            axis = (R - R.T) / (2 * np.sin(angle))
            axis_org = np.array([axis[2, 1], axis[0, 2], axis[1, 0]])
        else:
            axis_org = np.array([1,0,0])

        Q = np.zeros([4]);
        Q[0] = np.cos(angle/2)
        Q[1] = np.sin(angle/2) * axis_org[0]
        Q[2] = np.sin(angle/2) * axis_org[1]
        Q[3] = np.sin(angle/2) * axis_org[2]

        self.entry_quat_0.delete(0,99)
        self.entry_quat_1.delete(0,99)
        self.entry_quat_2.delete(0,99)
        self.entry_quat_3.delete(0,99)
        self.entry_quat_0.insert(0,Q[0])
        self.entry_quat_1.insert(0,Q[1])
        self.entry_quat_2.insert(0,Q[2])
        self.entry_quat_3.insert(0,Q[3])



    def resetbutton_pressed(self):
        """
        Event triggered function on the event of a push on the button Reset
        """
        self.M = np.array(
            [[ -1,  -1, 1],   #Node 0
            [ -1,   1, 1],    #Node 1
            [1,   1, 1],      #Node 2
            [1,  -1, 1],      #Node 3
            [-1,  -1, -1],    #Node 4
            [-1,  1, -1],     #Node 5
            [1,   1, -1],     #Node 6
            [1,  -1, -1]], dtype=float).transpose()
        
        rotM = np.eye(3);
        self.M = rotM.dot(self.M)
        self.setRotMatrix(rotM)
        #self.rotMToAngleAxis(rotM);
        #self.rotMToRotationVector(rotM);
        #self.rotMToEuler(rotM);
        #self.rotMToQuat(rotM);


        #AA
        self.entry_AA_angle.delete(0,99)
        self.entry_AA_angle.insert(0, 0)

        self.entry_AA_ax1.delete(0,99);
        self.entry_AA_ax2.delete(0,99);
        self.entry_AA_ax3.delete(0,99);
        self.entry_AA_ax1.insert(0, 1);
        self.entry_AA_ax2.insert(0, 0);
        self.entry_AA_ax3.insert(0, 0);

        #RV
        self.entry_rotV_1.delete(0,99);
        self.entry_rotV_2.delete(0,99);
        self.entry_rotV_3.delete(0,99);
        self.entry_rotV_1.insert(0, 0);
        self.entry_rotV_2.insert(0, 0);
        self.entry_rotV_3.insert(0, 0);

        #EA
        self.entry_EA_pitch.delete(0,99);
        self.entry_EA_yaw.delete(0,99);
        self.entry_EA_roll.delete(0,99);
        self.entry_EA_pitch.insert(0, 0);
        self.entry_EA_yaw.insert(0, 0);
        self.entry_EA_roll.insert(0, 0);


        #Reset quat
        self.entry_quat_0.delete(0,99)
        self.entry_quat_1.delete(0,99)
        self.entry_quat_2.delete(0,99)
        self.entry_quat_3.delete(0,99)
        self.entry_quat_0.insert(0,1)
        self.entry_quat_1.insert(0,0)
        self.entry_quat_2.insert(0,0)
        self.entry_quat_3.insert(0,0)


        self.rotM = rotM


        self.update_cube() #Update the cube
        pass

    
    def apply_AA(self):
        """
        Event triggered function on the event of a push on the button button_AA
        """
        #Example on hot to get values from entries:
       
        #self.resetbutton_pressed()
        self.M = np.array(
            [[ -1,  -1, 1],   #Node 0
            [ -1,   1, 1],    #Node 1
            [1,   1, 1],      #Node 2
            [1,  -1, 1],      #Node 3
            [-1,  -1, -1],    #Node 4
            [-1,  1, -1],     #Node 5
            [1,   1, -1],     #Node 6
            [1,  -1, -1]], dtype=float).transpose()

        angle = float(self.entry_AA_angle.get())
        axisX = float(self.entry_AA_ax1.get())
        axisY = float(self.entry_AA_ax2.get())
        axisZ = float(self.entry_AA_ax3.get())
        rotM = Eaa2rotM(angle, np.array([axisX, axisY, axisZ]))
    
        #self.M = rotM@self.M 
        
        
        self.M = np.dot(rotM, self.M)
        


        self.setRotMatrix(rotM)
        self.rotMToRotationVector(rotM);
        self.rotMToEuler(rotM);
        self.rotMToQuat(rotM);
        self.rotM = rotM
        self.update_cube()
        

    
    def apply_rotV(self):
        """
        Event triggered function on the event of a push on the button button_rotV 
        """
        #self.resetbutton_pressed()
        self.M = np.array(
            [[ -1,  -1, 1],   #Node 0
            [ -1,   1, 1],    #Node 1
            [1,   1, 1],      #Node 2
            [1,  -1, 1],      #Node 3
            [-1,  -1, -1],    #Node 4
            [-1,  1, -1],     #Node 5
            [1,   1, -1],     #Node 6
            [1,  -1, -1]], dtype=float).transpose()

        rV = np.array([[float(self.entry_rotV_1.get())],[float(self.entry_rotV_2.get())],[float(self.entry_rotV_3.get())]]);
        axisSacado = rV / np.linalg.norm(rV)
        angleSacado = np.linalg.norm(rV);

        rotM = Eaa2rotM(angleSacado, axisSacado)
        self.M = np.dot(rotM, self.M)
        self.setRotMatrix(rotM)
        self.rotMToAngleAxis(rotM);
        self.rotMToEuler(rotM);
        self.rotMToQuat(rotM);
        self.rotM = rotM
        self.update_cube()


        pass

    
    def apply_EA(self):
        """
        Event triggered function on the event of a push on the button button_EA
        """
        
        #self.resetbutton_pressed()
        self.M = np.array(
            [[ -1,  -1, 1],   #Node 0
            [ -1,   1, 1],    #Node 1
            [1,   1, 1],      #Node 2
            [1,  -1, 1],      #Node 3
            [-1,  -1, -1],    #Node 4
            [-1,  1, -1],     #Node 5
            [1,   1, -1],     #Node 6
            [1,  -1, -1]], dtype=float).transpose()
        
        rotM = eAngles2rotM(float(self.entry_EA_yaw.get()),float(self.entry_EA_pitch.get()),  float(self.entry_EA_roll.get()));
        self.M = np.dot(rotM, self.M)
        self.setRotMatrix(rotM)
        self.rotMToAngleAxis(rotM);
        self.rotMToRotationVector(rotM);
        self.rotMToQuat(rotM);
        self.rotM = rotM
        self.update_cube()



        pass

    
    def apply_quat(self):
        """
        Event triggered function on the event of a push on the button button_quat
        """

        #self.resetbutton_pressed()
        self.M = np.array(
            [[ -1,  -1, 1],   #Node 0
            [ -1,   1, 1],    #Node 1
            [1,   1, 1],      #Node 2
            [1,  -1, 1],      #Node 3
            [-1,  -1, -1],    #Node 4
            [-1,  1, -1],     #Node 5
            [1,   1, -1],     #Node 6
            [1,  -1, -1]], dtype=float).transpose()

        Q = np.zeros(4)
        Q[0] = float(self.entry_quat_0.get());
        Q[1] = float(self.entry_quat_1.get());
        Q[2] = float(self.entry_quat_2.get());
        Q[3] = float(self.entry_quat_3.get());

        Q = Q / np.linalg.norm(Q)

        rotM = quaternion_rotation_matrix(Q);
        self.M = np.dot(rotM, self.M)
        self.setRotMatrix(rotM)
        self.rotMToAngleAxis(rotM);
        self.rotMToRotationVector(rotM);
        self.rotMToEuler(rotM);
        self.rotM = rotM
        self.update_cube()



        pass

    
    def onclick(self, event):
        """
        Event triggered function on the event of a mouse click inside the figure canvas
        """
        print("Pressed button", event.button)

        if event.button:
            self.pressed = True # Bool to control(activate) a drag (click+move)

            x_fig,y_fig= self.canvas_coordinates_to_figure_coordinates(event.x,event.y) #Extract viewport coordinates
           
            radius = 20

            if (x_fig**2 + y_fig**2) < 0.5 * radius**2:
                Z = np.sqrt(radius**2 - x_fig**2 - y_fig**2)
                m0 = np.array([x_fig, y_fig, Z])
            else:
                Z = (radius**2) / (2 * np.sqrt(x_fig**2 + y_fig**2))
                m0 = np.array([x_fig, y_fig, Z]) / np.sqrt(x_fig**2 + y_fig**2 + Z**2)

            self.lastM = m0;
            


    def onmove(self,event):
        """
        Event triggered function on the event of a mouse motion
        """
        
        #Example
        if self.pressed: #Only triggered if previous click
            x_fig,y_fig= self.canvas_coordinates_to_figure_coordinates(event.x,event.y) #Extract viewport coordinates

            radius = 20

            # Assuming xmouse and ymouse are defined
            dist = x_fig*x_fig+y_fig*y_fig
            

            if dist < 40:

                if (x_fig**2 + y_fig**2) < 0.5 * radius**2:
                    Z = np.sqrt(radius**2 - x_fig**2 - y_fig**2)
                    m1 = np.array([x_fig, y_fig, Z])
                else:
                    Z = (radius**2) / (2 * np.sqrt(x_fig**2 + y_fig**2))
                    m1 = radius*(np.array([x_fig, y_fig, Z])) / np.sqrt(x_fig**2 + y_fig**2 + Z**2)


                axis = -np.cross(m1, self.lastM)
                axisNew = np.array([0,0,0]);
                axisNew[0] = -axis[1]
                axisNew[1] = axis[2]
                axisNew[2] = axis[0]
                # Obtain angle
                angle = np.rad2deg(np.arccos(np.dot(m1, self.lastM) / (np.linalg.norm(m1) * np.linalg.norm(self.lastM)))) * 1


                self.lastM = m1.copy()
                RotM = Eaa2rotM(angle, axisNew);

                

                self.M = RotM.dot(self.M)  # Modify the vertices matrix with a rotation matrix M
                self.rotM = RotM.dot(self.rotM);

                self.setRotMatrix(self.rotM)
                self.rotMToAngleAxis(self.rotM);
                self.rotMToRotationVector(self.rotM);
                self.rotMToEuler(self.rotM);
                self.rotMToQuat(self.rotM);
                self.update_cube()  # Update the cube


    def onrelease(self,event):
        """
        Event triggered function on the event of a mouse release
        """
        self.pressed = False # Bool to control(deactivate) a drag (click+move)


    def init_cube(self):
        """
        Initialization function that sets up cube's geometry and plot information
        """

        self.M = np.array(
            [[ -1,  -1, 1],   #Node 0
            [ -1,   1, 1],    #Node 1
            [1,   1, 1],      #Node 2
            [1,  -1, 1],      #Node 3
            [-1,  -1, -1],    #Node 4
            [-1,  1, -1],     #Node 5
            [1,   1, -1],     #Node 6
            [1,  -1, -1]], dtype=float).transpose() #Node 7

        self.con = [
            [0, 1, 2, 3], #Face 1
            [4, 5, 6, 7], #Face 2
            [3, 2, 6, 7], #Face 3
            [0, 1, 5, 4], #Face 4
            [0, 3, 7, 4], #Face 5
            [1, 2, 6, 5]] #Face 6

        faces = []

        for row in self.con:
            faces.append([self.M[:,row[0]],self.M[:,row[1]],self.M[:,row[2]],self.M[:,row[3]]])

        self.fig = plt.figure()
        ax = self.fig.add_subplot(111, projection='3d')

        for item in [self.fig, ax]:
            item.patch.set_visible(False)

        self.facesObj = Poly3DCollection(faces, linewidths=.2, edgecolors='k',animated = True)
        self.facesObj.set_facecolor([(0,0,1,0.9), #Blue
        (0,1,0,0.9), #Green
        (.9,.5,0.13,0.9), #Orange
        (1,0,0,0.9), #Red
        (1,1,0,0.9), #Yellow
        (0,0,0,0.9)]) #Black

        #Transfering information to the plot
        ax.add_collection3d(self.facesObj)

        #Configuring the plot aspect
        ax.azim=-90
        ax.roll = -90
        ax.elev=0   
        ax.set_xlim3d(-2, 2)
        ax.set_ylim3d(-2, 2)
        ax.set_zlim3d(-2, 2)
        ax.set_aspect('equal')
        ax.disable_mouse_rotation()
        ax.set_axis_off()

        self.pix2unit = 1.0/60 #ratio for drawing the cube 


    def update_cube(self):
        """
        Updates the cube vertices and updates the figure.
        Call this function after modifying the vertex matrix in self.M to redraw the cube
        """

        faces = []

        for row in self.con:
            faces.append([self.M[:,row[0]],self.M[:,row[1]],self.M[:,row[2]], self.M[:,row[3]]])

        self.facesObj.set_verts(faces)
        self.bm.update()


    def canvas_coordinates_to_figure_coordinates(self,x_can,y_can):
        """
        Remap canvas coordinates to cube centered coordinates
        """

        (canvas_width,canvas_height)=self.canvas.get_width_height()
        figure_center_x = canvas_width/2+14
        figure_center_y = canvas_height/2+2
        x_fig = (x_can-figure_center_x)*self.pix2unit
        y_fig = (y_can-figure_center_y)*self.pix2unit

        return(x_fig,y_fig)


    def destroy(self):
        """
        Close function to properly destroy the window and tk with figure
        """
        try:
            self.destroy()
        finally:
            exit()


class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
            cv.draw_idle()


if __name__ == "__main__":
    app = Arcball()
    app.mainloop()
    exit()

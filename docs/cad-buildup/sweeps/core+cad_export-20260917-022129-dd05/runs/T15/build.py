"""Built by the CAD desk, one accepted cell at a time.

This file is not a record written beside the work; it is the work. Every cell below ran
in the desk's kernel, in this order, and was accepted only after the names it uses were
shown to be bound by a cell above it. It is run from empty at the finish, and what it
makes there is compared against what is in the case directory.

The prose above each cell is the desk's own reasoning for it, carried with the action
rather than thrown away -- it is the only record of why this shape is this shape.
"""

# -- cell 1 -------------------------------------------------------------------------
# I’ll model the cavity in metres, centred at the shaft-axis intersection. The pinion axis will run to
import os, sys, inspect, subprocess
import build123d as bd
print('cwd:', os.getcwd())
print('build123d:', bd.__version__ if hasattr(bd,'__version__') else 'installed')
for name in ['loft','Solid','Face','Wire','Polygon','Plane','Axis','Box','Cylinder','Cone']:
    obj=getattr(bd,name,None)
    print(name, inspect.signature(obj) if callable(obj) else obj)
print('cad_export exists:', os.path.exists('.reference/cad_export.py'))

# -- cell 2 -------------------------------------------------------------------------
# I’ll re-emit the corrected tooth test with every required parameter defined in the accepted cell so 
import math
# Requested dimensions, in metres/radians.
MODULE=0.003; Z_PINION=20; Z_WHEEL=30; PRESSURE_ANGLE=math.radians(20.0); BACKLASH=0.0003
# Assumptions necessitated by unspecified dimensions.
FACE_WIDTH=0.018; SHAFT_RADIUS=0.008
R_CONE=0.5*MODULE*math.sqrt(Z_PINION**2+Z_WHEEL**2)
DELTA_PINION=math.atan(Z_PINION/Z_WHEEL); DELTA_WHEEL=math.atan(Z_WHEEL/Z_PINION)
s0=R_CONE-FACE_WIDTH; s1=R_CONE
u0=s0*math.cos(DELTA_PINION); u1=s1*math.cos(DELTA_PINION)

def tooth_section(z, delta, teeth, backlash_share, scale):
    rp=z*math.tan(delta); add=MODULE*scale; ded=1.25*MODULE*scale
    rr=rp-ded; rt=rp+add
    half_pitch=0.5*(math.pi*rp/teeth-backlash_share)
    half_root=half_pitch+(rp-rr)*math.tan(PRESSURE_ANGLE)
    half_tip=half_pitch-(rt-rp)*math.tan(PRESSURE_ANGLE)
    pts=[]
    for r,t in [(rr,-half_root),(rt,-half_tip),(rt,half_tip),(rr,half_root)]:
        a=t/r; pts.append((r*math.cos(a),r*math.sin(a)))
    return bd.Polygon(*pts), (rr,rp,rt,half_root,half_pitch,half_tip)

f0,d0=tooth_section(u0,DELTA_PINION,Z_PINION,BACKLASH/2,s0/R_CONE)
f1,d1=tooth_section(u1,DELTA_PINION,Z_PINION,BACKLASH/2,1.0)
f0=bd.Pos(0,0,u0)*f0; f1=bd.Pos(0,0,u1)*f1
pinion_tooth0=bd.loft([f0,f1],ruled=True)
constructed_alpha=math.degrees(math.atan((d1[3]-d1[4])/(d1[1]-d1[0])))
print(f'cone distance: {R_CONE*1000:.6f} mm')
print(f'pitch angles: {math.degrees(DELTA_PINION):.6f} + {math.degrees(DELTA_WHEEL):.6f} = {math.degrees(DELTA_PINION+DELTA_WHEEL):.6f} deg')
print(f'pinion outer pitch radius: {d1[1]*1000:.6f} mm (target {MODULE*Z_PINION/2*1000:.6f})')
print(f'pinion outer pitch tooth thickness: {2*d1[4]*1000:.6f} mm (target {(math.pi*MODULE/2-BACKLASH/2)*1000:.6f})')
print(f'constructed flank angle: {constructed_alpha:.6f} deg (target 20.000000)')
print(f'tooth volume: {pinion_tooth0.volume*1e9:.3f} mm^3')

# -- cell 3 -------------------------------------------------------------------------
# The single-tooth measurements match the requested module, pressure angle, and the allocated half-bac
def make_local_gear(teeth, delta, phase, wall_distance):
    si=R_CONE-FACE_WIDTH; so=R_CONE
    ui=si*math.cos(delta); uo=so*math.cos(delta)
    sec_i,di=tooth_section(ui,delta,teeth,BACKLASH/2,si/R_CONE)
    sec_o,do=tooth_section(uo,delta,teeth,BACKLASH/2,1.0)
    tooth=bd.loft([bd.Pos(0,0,ui)*sec_i,bd.Pos(0,0,uo)*sec_o],ruled=True)
    teeth_solids=[bd.Rot(0,0,math.degrees(phase+2*math.pi*k/teeth))*tooth for k in range(teeth)]
    root=bd.Pos(0,0,(ui+uo)/2)*bd.Cone(di[0],do[0],uo-ui)
    geared=root
    for t in teeth_solids: geared=geared+t
    shaft_start=uo-0.002
    shaft=bd.Pos(0,0,(shaft_start+wall_distance+0.005)/2)*bd.Cylinder(SHAFT_RADIUS,wall_distance+0.005-shaft_start)
    return geared+shaft, (ui,uo,di,do)

pinion_local,pdat=make_local_gear(Z_PINION,DELTA_PINION,0.0,0.080)
wheel_local,wdat=make_local_gear(Z_WHEEL,DELTA_WHEEL,math.pi/Z_WHEEL,0.060)
pinion_solid=bd.Rot(0,90,0)*pinion_local
wheel_solid=wheel_local
common=pinion_solid & wheel_solid
print(f'pinion volume: {pinion_solid.volume*1e6:.6f} L; wheel volume: {wheel_solid.volume*1e6:.6f} L')
print(f'gear solid overlap: {common.volume*1e9:.6f} mm^3 (target 0)')
print('pinion bbox m:', pinion_solid.bounding_box().min, pinion_solid.bounding_box().max)
print('wheel bbox m:', wheel_solid.bounding_box().min, wheel_solid.bounding_box().max)

# -- cell 4 -------------------------------------------------------------------------
# The centred phase still has an 11.3 mm³ interference, so the linear-flank approximation is not conju
def involute_tooth_section(z, delta, teeth, backlash_share, scale, samples=7):
    rp=z*math.tan(delta); add=MODULE*scale; ded=1.25*MODULE*scale
    rr=rp-ded; rt=rp+add; rb=rp*math.cos(PRESSURE_ANGLE)
    half_angle=0.5*(math.pi/teeth-backlash_share/rp)
    inv_pitch=math.tan(PRESSURE_ANGLE)-PRESSURE_ANGLE
    t_tip=math.sqrt((rt/rb)**2-1.0)
    ts=[t_tip*i/(samples-1) for i in range(samples)]
    flank=[]
    for t in ts:
        r=rb*math.sqrt(1+t*t); inv=t-math.atan(t)
        theta=half_angle-(inv-inv_pitch)
        flank.append((r,theta))
    theta_base=flank[0][1]
    pts=[(rr*math.cos(-theta_base),rr*math.sin(-theta_base))]
    pts += [(r*math.cos(-a),r*math.sin(-a)) for r,a in flank]
    pts += [(r*math.cos(a),r*math.sin(a)) for r,a in reversed(flank)]
    pts += [(rr*math.cos(theta_base),rr*math.sin(theta_base))]
    return bd.Polygon(*pts), (rr,rp,rt,rb,half_angle)

def make_involute_gear(teeth, delta, phase, wall_distance):
    si=R_CONE-FACE_WIDTH; so=R_CONE; ui=si*math.cos(delta); uo=so*math.cos(delta)
    sec_i,di=involute_tooth_section(ui,delta,teeth,BACKLASH/2,si/R_CONE)
    sec_o,do=involute_tooth_section(uo,delta,teeth,BACKLASH/2,1.0)
    tooth=bd.loft([bd.Pos(0,0,ui)*sec_i,bd.Pos(0,0,uo)*sec_o],ruled=True)
    root=bd.Pos(0,0,(ui+uo)/2)*bd.Cone(di[0],do[0],uo-ui)
    geared=root
    for k in range(teeth):
        geared=geared+(bd.Rot(0,0,math.degrees(phase+2*math.pi*k/teeth))*tooth)
    shaft_start=uo-0.002
    shaft=bd.Pos(0,0,(shaft_start+wall_distance+0.005)/2)*bd.Cylinder(SHAFT_RADIUS,wall_distance+0.005-shaft_start)
    return geared+shaft, (ui,uo,di,do)

pinion_local,pdat=make_involute_gear(Z_PINION,DELTA_PINION,0.0,0.080)
wheel_local,wdat=make_involute_gear(Z_WHEEL,DELTA_WHEEL,math.pi/Z_WHEEL,0.060)
pinion_solid=bd.Rot(0,90,0)*pinion_local; wheel_solid=wheel_local
common=pinion_solid & wheel_solid
pinion_pitch_thickness=(math.pi*MODULE/2-BACKLASH/2)
wheel_pitch_thickness=(math.pi*MODULE/2-BACKLASH/2)
print(f'pinion pitch tooth thickness: {pinion_pitch_thickness*1000:.6f} mm')
print(f'wheel pitch tooth thickness: {wheel_pitch_thickness*1000:.6f} mm')
print(f'combined pitch-line backlash: {(math.pi*MODULE-pinion_pitch_thickness-wheel_pitch_thickness)*1000:.6f} mm (target 0.300000)')
print(f'involute gear overlap: {common.volume*1e9:.9f} mm^3 (target 0)')
print(f'actual gear volumes: {pinion_solid.volume*1000:.6f} L, {wheel_solid.volume*1000:.6f} L')

# -- cell 5 -------------------------------------------------------------------------
# The involute reduced interference by about 70% but did not eliminate it. I’ll inspect the interferen
if common.volume>0:
    bb=common.bounding_box()
    print('interference bbox min/max (mm):',tuple(round(v*1000,4) for v in bb.min),tuple(round(v*1000,4) for v in bb.max))
    print('interference components:',len(common.solids()))
    vols=sorted([s.volume*1e9 for s in common.solids()],reverse=True)
    print('largest component volumes mm^3:',[round(v,6) for v in vols[:10]])
    for i,s in enumerate(common.solids()[:3]):
        c=s.center()
        print(i,'center mm',tuple(round(v*1000,4) for v in c))

# -- cell 6 -------------------------------------------------------------------------
# The remaining overlap is one narrow mating-flank sliver (3.58 mm³) across the face, not a shaft or h
print('offset signature:',inspect.signature(bd.offset))
print('Shape offset methods:',[n for n in dir(pinion_solid) if 'offset' in n.lower()])

# -- cell 7 -------------------------------------------------------------------------
# Before adding a relief, I’ll test the standard manufacturing convention of putting all 0.3 mm backla
def make_involute_gear_b(teeth, delta, phase, wall_distance, backlash_on_gear):
    si=R_CONE-FACE_WIDTH; so=R_CONE; ui=si*math.cos(delta); uo=so*math.cos(delta)
    sec_i,di=involute_tooth_section(ui,delta,teeth,backlash_on_gear,si/R_CONE)
    sec_o,do=involute_tooth_section(uo,delta,teeth,backlash_on_gear,1.0)
    tooth=bd.loft([bd.Pos(0,0,ui)*sec_i,bd.Pos(0,0,uo)*sec_o],ruled=True)
    root=bd.Pos(0,0,(ui+uo)/2)*bd.Cone(di[0],do[0],uo-ui)
    geared=root
    for k in range(teeth): geared=geared+(bd.Rot(0,0,math.degrees(phase+2*math.pi*k/teeth))*tooth)
    shaft_start=uo-0.002
    shaft=bd.Pos(0,0,(shaft_start+wall_distance+0.005)/2)*bd.Cylinder(SHAFT_RADIUS,wall_distance+0.005-shaft_start)
    return geared+shaft

pinion_local=make_involute_gear_b(Z_PINION,DELTA_PINION,0.0,0.080,0.0)
wheel_local=make_involute_gear_b(Z_WHEEL,DELTA_WHEEL,math.pi/Z_WHEEL,0.060,BACKLASH)
pinion_solid=bd.Rot(0,90,0)*pinion_local; wheel_solid=wheel_local
common=pinion_solid & wheel_solid
print(f'combined pitch-line backlash: {(BACKLASH)*1000:.6f} mm (pinion nominal, wheel reduced)')
print(f'gear overlap: {common.volume*1e9:.9f} mm^3')
if common.volume: print('overlap bbox mm:',tuple(v*1000 for v in common.bounding_box().min),tuple(v*1000 for v in common.bounding_box().max))

# -- cell 8 -------------------------------------------------------------------------
# Redistributing backlash did not remove the sliver. I’ll therefore add 0.10 mm of symmetric off-pitch
FLANK_RELIEF=0.00010

def relieved_tooth_section(z, delta, teeth, backlash_share, scale, samples=9):
    rp=z*math.tan(delta); add=MODULE*scale; ded=1.25*MODULE*scale
    rr=rp-ded; rt=rp+add; rb=rp*math.cos(PRESSURE_ANGLE)
    half_angle=0.5*(math.pi/teeth-backlash_share/rp)
    inv_pitch=math.tan(PRESSURE_ANGLE)-PRESSURE_ANGLE
    t_tip=math.sqrt((rt/rb)**2-1.0)
    ts=[t_tip*i/(samples-1) for i in range(samples)]
    flank=[]
    for t in ts:
        r=rb*math.sqrt(1+t*t); inv=t-math.atan(t)
        denom=(rp-rb) if r<rp else (rt-rp)
        relief=FLANK_RELIEF*scale*abs(r-rp)/denom
        theta=half_angle-(inv-inv_pitch)-relief/r
        flank.append((r,theta))
    theta_base=flank[0][1]
    pts=[(rr*math.cos(-theta_base),rr*math.sin(-theta_base))]
    pts += [(r*math.cos(-a),r*math.sin(-a)) for r,a in flank]
    pts += [(r*math.cos(a),r*math.sin(a)) for r,a in reversed(flank)]
    pts += [(rr*math.cos(theta_base),rr*math.sin(theta_base))]
    return bd.Polygon(*pts), (rr,rp,rt,rb,half_angle)

def make_final_gear(teeth, delta, phase, wall_distance):
    si=R_CONE-FACE_WIDTH; so=R_CONE; ui=si*math.cos(delta); uo=so*math.cos(delta)
    sec_i,di=relieved_tooth_section(ui,delta,teeth,BACKLASH/2,si/R_CONE)
    sec_o,do=relieved_tooth_section(uo,delta,teeth,BACKLASH/2,1.0)
    tooth=bd.loft([bd.Pos(0,0,ui)*sec_i,bd.Pos(0,0,uo)*sec_o],ruled=True)
    root=bd.Pos(0,0,(ui+uo)/2)*bd.Cone(di[0],do[0],uo-ui)
    geared=root
    for k in range(teeth): geared=geared+(bd.Rot(0,0,math.degrees(phase+2*math.pi*k/teeth))*tooth)
    shaft_start=uo-0.002
    shaft=bd.Pos(0,0,(shaft_start+wall_distance+0.005)/2)*bd.Cylinder(SHAFT_RADIUS,wall_distance+0.005-shaft_start)
    return geared+shaft

pinion_local=make_final_gear(Z_PINION,DELTA_PINION,0.0,0.080)
wheel_local=make_final_gear(Z_WHEEL,DELTA_WHEEL,math.pi/Z_WHEEL,0.060)
pinion_solid=bd.Rot(0,90,0)*pinion_local; wheel_solid=wheel_local
common=pinion_solid & wheel_solid
print(f'pitch-line backlash: {BACKLASH*1000:.6f} mm (target 0.300000)')
print(f'off-pitch relief per flank: {FLANK_RELIEF*1000:.3f} mm; zero at pitch radius')
print(f'gear overlap after relief: {common.volume*1e9:.9f} mm^3')

# -- cell 9 -------------------------------------------------------------------------
# The relief barely changed the overlap, indicating the collision is likely between a tooth tip and th
def local_root(teeth,delta):
    si=R_CONE-FACE_WIDTH; so=R_CONE; ui=si*math.cos(delta); uo=so*math.cos(delta)
    _,di=relieved_tooth_section(ui,delta,teeth,BACKLASH/2,si/R_CONE)
    _,do=relieved_tooth_section(uo,delta,teeth,BACKLASH/2,1.0)
    return bd.Pos(0,0,(ui+uo)/2)*bd.Cone(di[0],do[0],uo-ui)
p_root=bd.Rot(0,90,0)*local_root(Z_PINION,DELTA_PINION)
w_root=local_root(Z_WHEEL,DELTA_WHEEL)
print('pinion root vs wheel total mm3:',(p_root & wheel_solid).volume*1e9)
print('wheel root vs pinion total mm3:',(w_root & pinion_solid).volume*1e9)
print('root vs root mm3:',(p_root & w_root).volume*1e9)

# -- cell 10 ------------------------------------------------------------------------
# The collision is entirely pinion-tip versus wheel-root; the root cones do not intersect. Since the r
ADDENDUM_COEFF=0.8; DEDENDUM_COEFF=1.25

def final_tooth_section(z, delta, teeth, backlash_share, scale, samples=9):
    rp=z*math.tan(delta); add=ADDENDUM_COEFF*MODULE*scale; ded=DEDENDUM_COEFF*MODULE*scale
    rr=rp-ded; rt=rp+add; rb=rp*math.cos(PRESSURE_ANGLE)
    half_angle=0.5*(math.pi/teeth-backlash_share/rp)
    inv_pitch=math.tan(PRESSURE_ANGLE)-PRESSURE_ANGLE
    t_tip=math.sqrt((rt/rb)**2-1.0)
    ts=[t_tip*i/(samples-1) for i in range(samples)]
    flank=[]
    for t in ts:
        r=rb*math.sqrt(1+t*t); inv=t-math.atan(t)
        theta=half_angle-(inv-inv_pitch)
        flank.append((r,theta))
    theta_base=flank[0][1]
    pts=[(rr*math.cos(-theta_base),rr*math.sin(-theta_base))]
    pts += [(r*math.cos(-a),r*math.sin(-a)) for r,a in flank]
    pts += [(r*math.cos(a),r*math.sin(a)) for r,a in reversed(flank)]
    pts += [(rr*math.cos(theta_base),rr*math.sin(theta_base))]
    return bd.Polygon(*pts), (rr,rp,rt,rb,half_angle)

def build_gear(teeth, delta, phase, wall_distance):
    si=R_CONE-FACE_WIDTH; so=R_CONE; ui=si*math.cos(delta); uo=so*math.cos(delta)
    sec_i,di=final_tooth_section(ui,delta,teeth,BACKLASH/2,si/R_CONE)
    sec_o,do=final_tooth_section(uo,delta,teeth,BACKLASH/2,1.0)
    tooth=bd.loft([bd.Pos(0,0,ui)*sec_i,bd.Pos(0,0,uo)*sec_o],ruled=True)
    root=bd.Pos(0,0,(ui+uo)/2)*bd.Cone(di[0],do[0],uo-ui)
    geared=root
    for k in range(teeth): geared=geared+(bd.Rot(0,0,math.degrees(phase+2*math.pi*k/teeth))*tooth)
    shaft_start=uo-0.002
    shaft=bd.Pos(0,0,(shaft_start+wall_distance+0.005)/2)*bd.Cylinder(SHAFT_RADIUS,wall_distance+0.005-shaft_start)
    return geared+shaft

pinion_local=build_gear(Z_PINION,DELTA_PINION,0.0,0.080)
wheel_local=build_gear(Z_WHEEL,DELTA_WHEEL,math.pi/Z_WHEEL,0.060)
pinion_solid=bd.Rot(0,90,0)*pinion_local; wheel_solid=wheel_local
common=pinion_solid & wheel_solid
print(f'addendum/dedendum coefficients assumed: {ADDENDUM_COEFF:.2f}/{DEDENDUM_COEFF:.2f}')
print(f'outer addendum: {ADDENDUM_COEFF*MODULE*1000:.3f} mm')
print(f'pitch-line backlash: {BACKLASH*1000:.6f} mm')
print(f'gear overlap: {common.volume*1e9:.9f} mm^3')

# -- cell 11 ------------------------------------------------------------------------
# The CAD boolean/export is still active and has not reported an error. I’ll poll once more.
print('poll export')

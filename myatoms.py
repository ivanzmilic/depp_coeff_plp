from lightweaver.atomic_model import *
from lightweaver.collisional_rates import *
from lightweaver.broadening import *
from lightweaver.atomic_table import Element

Fe23_5250 = lambda: \
AtomicModel(element=Element(Z=26),
	levels=[
		AtomicLevel(E=     0.000, g=9, label="FeI 3d6 4s2   a5DE 4", stage=0, J=Fraction(4, 1), L=2, S=Fraction(2, 1)),
		AtomicLevel(E=   415.932, g=7, label="FeI 3d6 4s2   a5DE 3", stage=0, J=Fraction(3, 1), L=2, S=Fraction(2, 1)),
		AtomicLevel(E=   704.004, g=5, label="FeI 3d6 4s2   a5DE 2", stage=0, J=Fraction(2, 1), L=2, S=Fraction(2, 1)),
		AtomicLevel(E=   888.129, g=3, label="FeI 3d6 4s2   a5DE 1", stage=0, J=Fraction(1, 1), L=2, S=Fraction(2, 1)),
		AtomicLevel(E=   978.074, g=1, label="FeI 3d6 4s2   a5DE 0", stage=0, J=Fraction(0, 1), L=2, S=Fraction(2, 1)),
		AtomicLevel(E=  6928.266, g=35, label="FeI 3d7 4s      a5FE", stage=0, J=Fraction(17, 1), L=3, S=Fraction(2, 1)),
		AtomicLevel(E= 17550.176, g=15, label="FeI 3d7 4s      a5PE", stage=0, J=Fraction(7, 1), L=1, S=Fraction(2, 1)),
		AtomicLevel(E= 19350.893, g=11, label="FeI 3d6 4s 4p z7DO 5", stage=0, J=Fraction(5, 1), L=2, S=Fraction(3, 1)),
		AtomicLevel(E= 19562.439, g=9, label="FeI 3d6 4s 4p z7DO 4", stage=0, J=Fraction(4, 1), L=2, S=Fraction(3, 1)),
		AtomicLevel(E= 19757.033, g=7, label="FeI 3d6 4s 4p z7DO 3", stage=0, J=Fraction(3, 1), L=2, S=Fraction(3, 1)),
		AtomicLevel(E= 19912.494, g=5, label="FeI 3d6 4s 4p z7DO 2", stage=0, J=Fraction(2, 1), L=2, S=Fraction(3, 1)),
		AtomicLevel(E= 20019.635, g=3, label="FeI 3d6 4s 4p z7DO 1", stage=0, J=Fraction(1, 1), L=2, S=Fraction(3, 1)),
		AtomicLevel(E= 25899.986, g=25, label="FeI 3d6 4s 4p   z5DO", stage=0, J=Fraction(12, 1), L=2, S=Fraction(2, 1)),
		AtomicLevel(E= 26874.549, g=35, label="FeI 3d6 4s 4p   z5FO", stage=0, J=Fraction(17, 1), L=3, S=Fraction(2, 1)),
		AtomicLevel(E= 29056.320, g=7, label="FeI 3d6 4s 4p z5PO 3", stage=0, J=Fraction(3, 1), L=1, S=Fraction(2, 1)),
		AtomicLevel(E= 29469.020, g=5, label="FeI 3d6 4s 4p z5PO 2", stage=0, J=Fraction(2, 1), L=1, S=Fraction(2, 1)),
		AtomicLevel(E= 29732.732, g=3, label="FeI 3d6 4s 4p z5PO 1", stage=0, J=Fraction(1, 1), L=1, S=Fraction(2, 1)),
		AtomicLevel(E= 44677.004, g=9, label="FeI 3d6 4s 5s e5DE 4", stage=0, J=Fraction(4, 1), L=2, S=Fraction(2, 1)),
		AtomicLevel(E= 45061.328, g=7, label="FeI 3d6 4s 5s e5DE 3", stage=0, J=Fraction(3, 1), L=2, S=Fraction(2, 1)),
		AtomicLevel(E= 45333.875, g=5, label="FeI 3d6 4s 5s e5DE 2", stage=0, J=Fraction(2, 1), L=2, S=Fraction(2, 1)),
		AtomicLevel(E= 45509.148, g=3, label="FeI 3d6 4s 5s e5DE 1", stage=0, J=Fraction(1, 1), L=2, S=Fraction(2, 1)),
		AtomicLevel(E= 45595.078, g=1, label="FeI 3d6 4s 5s e5DE 0", stage=0, J=Fraction(0, 1), L=2, S=Fraction(2, 1)),
		AtomicLevel(E= 63737.000, g=50, label="FeII 3d6 4s     a6DE", stage=1, J=Fraction(49, 2), L=2, S=Fraction(5, 2)),
	],
	lines=[
		VoigtLine(j=11, i=4, f=1.150e-05, type=LineType.CRD, quadrature=LinearCoreExpWings(qCore=1, qWing=4, Nlambda=81), broadening=LineBroadening(natural=[RadiativeBroadening(gamma=930)], elastic=[VdwUnsold(vals=[2.0, 2.0]), MultiplicativeStarkBroadening(coeff=0)])),
	],
	continua=[		
	],
	collisions=[
	])
